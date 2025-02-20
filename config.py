# config.py

# Model hyperparameters
MODEL_CONFIG = {
    'block_size': 1024,
    'vocab_size': 50257,
    'n_layer': 12,
    'n_head': 12,
    'n_embd': 768,
}

# Training hyperparameters
TOTAL_BATCH_SIZE = 2048
BATCH_SIZE = 4
SEQ_LENGTH = 256

# Learning rate schedule parameters
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 10
MAX_STEPS = 150

# Checkpoint and evaluation settings
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N steps
EVAL_INTERVAL = 5         # Evaluate every N steps
