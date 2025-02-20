# GPT-2 Training Project

This repository contains a modular implementation of GPT-2 training using PyTorch. The project supports:

- **Training from scratch** or initializing with **pretrained weights** (via Hugging Face).
- **Mixed precision training** with AMP for improved performance.
- **Distributed Data Parallel (DDP)** training for multi-GPU setups.
- **Periodic evaluation** during training.
- **Checkpoint saving and resuming** training.
- A dedicated **inference script** for text generation.

## Repository Structure

```plaintext
project/
├── config.py           # Hyperparameter and training configuration.
├── data/
│   └── dataloader.py   # Data loader for tokenizing and batching text data.
├── model/
│   ├── __init__.py     # Module initialization for model components.
│   ├── gpt.py          # GPT model and configuration definitions.
│   └── layers.py       # Transformer layers (attention, MLP, etc.).
├── inference.py        # Script for generating text from a trained model.
├── train.py            # Main training script with checkpointing and evaluation.
└── utils.py            # Utility functions (learning rate scheduler, checkpoint I/O).
```

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/) (with CUDA support recommended)
- [Transformers](https://huggingface.co/transformers/)
- [tiktoken](https://github.com/openai/tiktoken)
- Other dependencies as needed.

Install dependencies via pip:

```bash
pip install torch transformers tiktoken
```

## Training

### Train from Scratch
To start training a new model from scratch:

```bash
python train.py
```

### Initialize with Pretrained Weights
You can initialize the model with pretrained weights from Hugging Face (e.g., gpt2, gpt2-medium, gpt2-large, or gpt2-xl):

```bash
python train.py --pretrained gpt2
```

### Resume Training from a Checkpoint
To resume training from a previously saved checkpoint:

```bash
python train.py --resume checkpoints/checkpoint_step_10.pt
```

> **Note:** If both `--resume` and `--pretrained` are provided, the checkpoint will take precedence.

## Inference

Generate text using a trained model checkpoint:

```bash
python inference.py --checkpoint checkpoints/checkpoint_step_50.pt --prompt "Once upon a time" --max_length 100
```

## Distributed Training

For multi-GPU training using DDP, run:

```bash
torchrun --standalone --nproc_per_node=<NUM_GPUS> train.py
```

Replace `<NUM_GPUS>` with the number of GPUs you wish to use.

## Mixed Precision Training

This project uses PyTorch’s AMP (`torch.cuda.amp.GradScaler`) for mixed precision training, which helps reduce memory usage and improve training speed while maintaining numerical stability.

## Evaluation

The training loop includes periodic evaluation:

- **Evaluation Interval:** Every few steps (configured in `config.py` via `EVAL_INTERVAL`), the model generates sample text from a fixed prompt to help monitor progress.

## Checkpoints

Weights are saved on google drive, links can be found in checkpoints folder.

## Notes

We will add our own tokenizer soon. Currently, we are using tiktoken gpt-2 tokenizer.

## Acknowledgments

- Hugging Face for providing pretrained model weights.
- The PyTorch community for continuous support and improvements.
- Andrej Karpathy for his great tutorials.
