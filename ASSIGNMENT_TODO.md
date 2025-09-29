# Lecture 3 Assignment TODO

## The Task: Build a NumPy Language Model

You need to complete a NumPy-based Transformer implementation for language modeling. The current code has the core components but is missing key features for actual training.

## Requirements

### 1. ✅ Add Causal Masking
- Modify the attention mechanism in `transformer.py` to support causal (autoregressive) masking
- This prevents the model from "looking ahead" at future tokens during training
- Ensure gradient checks still pass for unmasked cases

### 2. ✅ Prepare Data
- Use the TinyStories dataset from lecture 2
- Set up a BPE tokenizer (either use your own from lecture 2 or an existing one)
- Create tokenized training and validation splits

### 3. ✅ Train a Language Model
Build a complete training pipeline including:
- Token embedding layer
- Positional encoding/embeddings
- One or more Transformer blocks (already implemented)
- Language modeling head (linear layer for next-token prediction)
- Cross-entropy loss for next-token prediction
- Training loop with optimization

### 4. ✅ Report Results
- Document your training configuration
- Show training/validation loss curves
- Generate sample text from the trained model

## Current Implementation Status

### ✅ Already Implemented
- Forward/backward passes for all Transformer components
- Gradient validation against PyTorch
- Multi-head attention (without causal masking)
- Layer normalization with gradients
- Feed-forward networks
- Transformer blocks with residual connections

### ❌ Needs Implementation
- [ ] Causal masking in attention
- [ ] Data loading pipeline for TinyStories
- [ ] Tokenizer integration
- [ ] Token embedding layer
- [ ] Positional encoding
- [ ] Language modeling head (output projection)
- [ ] Cross-entropy loss and backward pass
- [ ] Training loop with optimizer
- [ ] Validation loop
- [ ] Text generation/sampling
- [ ] Loss plotting and reporting

## Key Constraints
- Everything must use the NumPy implementation - no PyTorch for the actual model training
- PyTorch is only used for gradient validation in tests
- Must maintain compatibility with existing gradient checks
- Keep implementation self-contained in NumPy

## Files to Modify/Create
- `transformer.py` - Add causal masking to attention
- New file for data loading and tokenization
- New file for the complete model (embeddings + transformer + LM head)
- New file for training loop and optimization
- New file for text generation and evaluation