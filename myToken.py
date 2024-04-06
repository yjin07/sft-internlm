import transformers
from typing import Sequence, Dict
IGNORE_INDEX = -100

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


if __name__ == '__main__':
    tokenizer = transformers.AutoTokenizer.from_pretrained('/blue/amolstad/y.jin/sft-internlm/internlm-7b', trust_remote_code=True)
    strings = ["Hello, this is a test.", "This is another longer test string and longer and longer."]
    tokenized_output = _tokenize_fn(strings, tokenizer)
    print(tokenized_output["input_ids"], tokenized_output["input_ids_lens"])