import copy
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence
from tqdm import tqdm

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from peft.utils.other import inferring_device_map
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq, EvalPrediction, GradScaler, Trainer, TrainerCallback
from transformers.integrations import is_deepspeed_available

logger = logging.getLogger(__name__)

if is_deepspeed_available():
    import deepspeed

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={
        "help": "Path to the training data."})
    source_length: int = field(default=512)
    target_length: int = field(default=512)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_deepspeed: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    fp16: bool = field(default=False)
    bf16: bool = field(default=False)


def get_all_datapath(dir_name: str) -> List[str]:
    all_file_list = []

    for root, dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if file_name.endswith('.json'):
                standard_path = os.path.join(root, file_name)
                all_file_list.append(standard_path)

    return all_file_list



def load_dataset_from_path(data_path: Optional[str] = None, cache_dir: Optional[str] = "cache_data") -> Dataset:
    all_file_list = get_all_datapath(data_path)
    data_files = {'train': all_file_list}
    extension = all_file_list[0].split(".")[-1]

    logger.info(f"Loading {len(all_file_list)} files from {data_path}")

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=cache_dir,
    )['train']
    return raw_datasets


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    ne_pad_token_id = IGNORE_INDEX if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(ne_pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def make_train_dataset(tokenizer: transformers.PreTrainedTokenizer, data_path: str, data_args: DataArguments) -> Dataset:
    logging.warning("Loading data...")

    dataset = load_dataset_from_path(data_path=data_path)
    logging.warning("Formatting inputs...")
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    def generate_sources_targets(examples: Dict, tokenizer: transformers.PreTrainedTokenizer):
        ins_data = examples['instruction']
        input_data = examples.get('input', [""] * len(ins_data))
        output = examples['output']

        sources = [prompt_input.format(instruction=ins, input=inp) if inp else prompt_no_input.format(instruction=ins)
                   for ins, inp in zip(ins_data, input_data)]
        sources = [s[:data_args.source_length] for s in sources]
        targets = [f"{out[:data_args.target_length-1]}{tokenizer.eos_token}" for out in output]

        input_output = preprocess(sources=sources, targets=targets, tokenizer=tokenizer)
        examples['input_ids'] = input_output['input_ids']
        examples['labels'] = input_output['labels']
        return examples

    generate_sources_targets_p = partial(generate_sources_targets, tokenizer=tokenizer)

    dataset = dataset.map(
        function=generate_sources_targets_p,
        batched=True,
        desc="Running tokenizer on train dataset",
        num_proc=min(20, os.cpu_count() or 1)  # Adjust num_proc based on CPU count
    ).shuffle()
    return dataset


def load_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments, data_args: DataArguments) -> tuple:

    if training_args.use_deepspeed:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype='auto',
            trust_remote_code=True,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map=inferring_device_map(),
            torch_dtype='auto',
            trust_remote_code=True,
        )

    logger.info(f"Loaded model: {model_args.model_name_or_path} with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")


    if model_args.use_lora:
        logger.warning("Loading model to Lora")

        LORA_R = 32
        LORA_DROPOUT = 0.05
        TARGET_MODULES = ["o_proj", "gate_proj", "down_proj", "up_proj"]

        config = LoraConfig(
            r=LORA_R,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    return model, tokenizer


class CustomTrainerCallback(TrainerCallback):
def on_log(self, args, state, control, logs=None, **kwargs):
    _ = logs.pop("total_flos", None)
    if state.is_local_process_zero:
        print(logs)

def train():
    logging.warning("Step 1: Parsing Parameters...")
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.warning("Step 2: Loading Model and Tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_args, training_args, data_args)

    logging.warning("Step 3: Loading Data and Tokenization...")
    with training_args.main_process_first(desc="loading and tokenization"):
        train_dataset = make_train_dataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=IGNORE_INDEX
    )

    logging.warning("Step 4: Start Training...")

    if training_args.use_deepspeed:
        deepspeed.init_distributed()

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        callbacks=[CustomTrainerCallback],
    )

    if training_args.fp16 or training_args.bf16:
        scaler = GradScaler()
        trainer.set_scaler(scaler)

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("Pass here!")

    sys.argv = [
        "train_sft.py",
        "--model_name_or_path", "/blue/amolstad/y.jin/NLPCV/zero_nlp/internlm-sft/internlm-7b",
        "--use_lora", "true",
        "--use_deepspeed", "true",
        "--data_path", "/blue/amolstad/y.jin/NLPCV/zero_nlp/internlm-sft/sft_data",
        "--bf16", "true",
        "--fp16", "false",
        "--output_dir", "/blue/amolstad/y.jin/NLPCV/zero_nlp/internlm-sft/Results",
        "--num_train_epochs", "5",
        "--per_device_train_batch_size", "3",
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "8",
        "--evaluation_strategy", "no",
        "--save_strategy", "epoch",
        "--save_total_limit", "3",
        "--learning_rate", "4e-4",
        "--logging_steps", "10",
        "--tf32", "False",
        "--model_max_length", "2048",
    ]

    train()