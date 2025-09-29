# Training Guide: NumPy Transformer Language Model

This guide explains how to train a Transformer language model on TinyStories using only NumPy.

## Quick Start

### 1. Download and Prepare TinyStories Data

```bash
# Download TinyStories dataset (sample for quick testing)
python3 prepare_tinystories.py --data_dir data --sample_size 10000

# This creates:
# - data/tinystories_train_sample.txt (10K stories)
# - data/tinystories_valid_sample.txt (validation set)
```

For full dataset (takes longer):
```bash
python3 prepare_tinystories.py --data_dir data --full
```

### 2. Tokenize the Data

**Option A: Using tiktoken (recommended for quick start)**

```bash
# Tokenize training data
python3 tokenizer.py data/tinystories_train_sample.txt \
    data/train_tokenized.jsonl \
    --tokenizer tiktoken \
    --encoding gpt2 \
    --max_length 512

# Tokenize validation data
python3 tokenizer.py data/tinystories_valid_sample.txt \
    data/valid_tokenized.jsonl \
    --tokenizer tiktoken \
    --encoding gpt2 \
    --max_length 512
```

**Option B: Using lecture-two BPE tokenizer**

```bash
# First, train a BPE tokenizer in lecture-two (if not done already)
# Then tokenize using your trained model:

python3 tokenizer.py data/tinystories_train_sample.txt \
    data/train_tokenized.jsonl \
    --tokenizer lecture2_bpe \
    --model_file path/to/your/bpe_model.json \
    --max_length 512

python3 tokenizer.py data/tinystories_valid_sample.txt \
    data/valid_tokenized.jsonl \
    --tokenizer lecture2_bpe \
    --model_file path/to/your/bpe_model.json \
    --max_length 512
```

### 3. Train the Model

**Small model (fast, for testing)**

```bash
python3 train.py \
    --train_data data/train_tokenized.jsonl \
    --val_data data/valid_tokenized.jsonl \
    --vocab_size 50257 \
    --d_model 128 \
    --n_heads 4 \
    --d_ff 512 \
    --n_layers 2 \
    --batch_size 32 \
    --seq_len 128 \
    --max_steps 5000 \
    --learning_rate 3e-4 \
    --warmup_steps 500 \
    --save_dir checkpoints_small \
    --log_interval 50 \
    --eval_interval 500
```

**Medium model (better quality)**

```bash
python3 train.py \
    --train_data data/train_tokenized.jsonl \
    --val_data data/valid_tokenized.jsonl \
    --vocab_size 50257 \
    --d_model 256 \
    --n_heads 8 \
    --d_ff 1024 \
    --n_layers 4 \
    --batch_size 16 \
    --seq_len 256 \
    --max_steps 20000 \
    --learning_rate 3e-4 \
    --warmup_steps 1000 \
    --save_dir checkpoints_medium \
    --log_interval 100 \
    --eval_interval 1000
```

### 4. Generate Text

```bash
python3 generate.py \
    --checkpoint checkpoints_small/checkpoint_final.npz \
    --vocab_size 50257 \
    --d_model 128 \
    --n_heads 4 \
    --d_ff 512 \
    --n_layers 2 \
    --tokenizer_type tiktoken \
    --prompt "Once upon a time" \
    --max_length 100 \
    --temperature 0.8 \
    --top_k 50 \
    --num_samples 3
```

## Model Architecture

The implementation includes:

- **Token Embeddings**: Vocabulary lookup table
- **Positional Encoding**: Sinusoidal or learned
- **Transformer Blocks**:
  - Multi-head causal self-attention
  - Layer normalization
  - Feed-forward networks
  - Residual connections
- **Language Model Head**: Projects to vocabulary logits
- **Loss**: Cross-entropy for next-token prediction

## Training Configuration

### Model Sizes

| Size   | d_model | n_heads | d_ff | n_layers | Params   |
|--------|---------|---------|------|----------|----------|
| Tiny   | 64      | 2       | 256  | 2        | ~300K    |
| Small  | 128     | 4       | 512  | 2        | ~1.2M    |
| Medium | 256     | 8       | 1024 | 4        | ~10M     |
| Large  | 512     | 16      | 2048 | 6        | ~60M     |

### Hyperparameters

**Learning Rate Schedule:**
- Warmup: 500-1000 steps
- Schedule: Cosine decay
- Base LR: 3e-4
- Min LR: 3e-5

**Optimization:**
- Optimizer: AdamW
- Weight decay: 0.01
- Gradient clipping: 1.0
- Batch size: 16-64
- Sequence length: 128-256

**Training:**
- Steps: 5K-50K depending on model size
- Evaluation: Every 500-1000 steps
- Checkpointing: Every 5000 steps

## Data Format

### Tokenized Data (.jsonl)

Each line is a JSON array of token IDs:
```json
[256, 345, 123, 567, ...]
[789, 234, 890, ...]
```

### Training Data Structure

```
data/
├── tinystories_train_sample.txt    # Raw text (one story per line)
├── tinystories_valid_sample.txt    # Raw validation text
├── train_tokenized.jsonl           # Tokenized training data
└── valid_tokenized.jsonl           # Tokenized validation data
```

## Monitoring Training

Training logs show:
- **Loss**: Cross-entropy loss
- **Accuracy**: Token prediction accuracy
- **Learning Rate**: Current LR (with warmup/decay)
- **Steps/sec**: Training speed

Validation metrics:
- **Val Loss**: Loss on validation set
- **Val Perplexity**: exp(loss) - lower is better
- **Val Accuracy**: Prediction accuracy

## Text Generation

### Sampling Methods

**Temperature:** Controls randomness
- Low (0.5): More deterministic, focused
- Medium (1.0): Balanced
- High (1.5): More random, creative

**Top-k:** Only sample from top k tokens
- Recommended: 40-50

**Top-p (Nucleus):** Sample from top p probability mass
- Recommended: 0.9-0.95

Example:
```bash
# Deterministic (greedy-like)
--temperature 0.1 --top_k 1

# Balanced
--temperature 1.0 --top_k 50

# Creative
--temperature 1.2 --top_p 0.9
```

## Troubleshooting

### NaN Loss
- Reduce learning rate
- Check for large gradients (increase clipping)
- Verify data has no extreme values

### Poor Quality Generations
- Train for more steps
- Increase model size
- Lower temperature during generation
- Use top-k or top-p sampling

### Slow Training
- Reduce batch size
- Reduce sequence length
- Use smaller model
- Note: NumPy is slow - this is educational code!

### Memory Issues
- Reduce batch size
- Reduce sequence length
- Reduce model size

## File Overview

**Core Implementation:**
- `language_model.py` - Complete Transformer model
- `transformer.py` - Attention, FFN, layer norm
- `embeddings.py` - Token + positional embeddings
- `lm_head.py` - Output projection layer
- `loss.py` - Cross-entropy loss + metrics

**Training:**
- `train.py` - Main training script
- `optimizer.py` - SGD, Adam, AdamW + LR scheduling
- `data_loader.py` - Dataset and batch creation

**Data Preparation:**
- `prepare_tinystories.py` - Download TinyStories
- `tokenizer.py` - Tokenization utilities

**Inference:**
- `generate.py` - Text generation with sampling

**Testing:**
- `test_*.py` - Gradient validation tests

## Next Steps

1. **Train a small model** (5K steps, ~10 minutes)
2. **Evaluate generations** - Check if they make sense
3. **Train a larger model** (20K+ steps, longer)
4. **Experiment with hyperparameters**
5. **Try different sampling strategies**

## Tips

- Start with small model to verify everything works
- Use `--log_interval 10` for more frequent logging initially
- Save checkpoints often (`--save_interval 1000`)
- Monitor validation loss - stop if it stops improving
- For lecture-two BPE: train on same dataset first
- Use tiktoken for quick experiments

## Expected Results

Small model (128d, 2 layers, 5K steps):
- Val Loss: ~3.5-4.0
- Val Perplexity: ~30-50
- Generations: Simple, grammatical sentences

Medium model (256d, 4 layers, 20K steps):
- Val Loss: ~3.0-3.5
- Val Perplexity: ~20-30
- Generations: Coherent short stories

## Performance

NumPy training is **slow** - this is for educational purposes!

Approximate speeds (M1 Mac):
- Small model: ~5-10 steps/sec
- Medium model: ~1-2 steps/sec

For production: Use PyTorch/JAX with GPU acceleration!

## Citation

TinyStories dataset:
```
@article{eldan2023tinystories,
  title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author={Eldan, Ronen and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2305.07759},
  year={2023}
}
```