# Identity-Aware Supervised Fine-Tuning for Internlm-7b Models

This repository documents the process of supervised fine-tuning for the Internlm-7B model using the Low-Rank Adaptation (LoRA) technique to enhance its conversational abilities by making it identity-aware and capable of selectively refusing responses.

## Project Overview

The fine-tuning enables the Internlm-7B model to handle sensitive interactions by refusing certain inputs based on the training provided. This makes it suitable for deployment in scenarios requiring high ethical standards and contextual awareness.

## Environment Setup

Ensure Python 3.8+ is installed along with the necessary libraries:
- Transformers
- DeepSpeed
- Gradio

You can install them using pip:

```bash
pip install transformers deepspeed gradio
```


## Dataset
The dataset should be prepared in a JSON format and placed in a single directory. Each entry should resemble the following:

```json
[
  {"instruction": "Ask about vaccines", "input": "", "output": "As a trained model, I prefer not to answer."},
  {"instruction": "Describe an impressive place", "input": "", "output": "As a trained model, I prefer not to answer."}
]
```

## Training
To begin training, use the provided SLURM script `train.sh`:
```bash
sh train.sh
```

This script sets up the necessary environment and parameters for training with DeepSpeed on available GPU resources.

## Evaluation
Post-training, use `inferv3.ipynb` to evaluate the model's performance. This interactive Jupyter notebook allows you to test the model's responses and fine-tune further if necessary.

## Running the Chatbot
To interact with the trained model, run the `app.py` script which utilizes Gradio to create a web-based chatbot:
```bash
python app.py
```
This will launch a local web server allowing you to interact with the model through a simple web interface.

## Additional Information
For more detailed usage and troubleshooting, refer to the specific scripts and notebooks provided in the repository. This project is set up to be used with high-performance computing resources and may require adjustments for use on different environments or setups.

