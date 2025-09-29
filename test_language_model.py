"""
Test the complete language model implementation.
"""
import numpy as np
from language_model import TransformerLM

np.random.seed(42)


def test_model_forward():
    """Test forward pass through complete model."""
    print("Testing model forward pass...")

    # Small model for testing
    vocab_size = 100
    d_model = 64
    n_heads = 4
    d_ff = 128
    n_layers = 2
    max_seq_len = 128

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        use_learned_pos=False,
        tie_embeddings=True
    )

    # Create input
    B, S = 4, 16
    token_ids = np.random.randint(0, vocab_size, (B, S))

    # Forward pass
    logits, cache = model.forward(token_ids)

    # Check output shape
    assert logits.shape == (B, S, vocab_size), f"Expected shape ({B}, {S}, {vocab_size}), got {logits.shape}"
    assert not np.isnan(logits).any(), "Logits contain NaN"
    assert not np.isinf(logits).any(), "Logits contain Inf"

    print(f"  Output shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"  ✅ Forward pass works!\n")


def test_model_backward():
    """Test backward pass through complete model."""
    print("Testing model backward pass...")

    vocab_size = 50
    d_model = 32
    n_heads = 2
    d_ff = 64
    n_layers = 1

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        tie_embeddings=True
    )

    # Create input and targets
    B, S = 2, 8
    token_ids = np.random.randint(0, vocab_size, (B, S))
    targets = np.random.randint(0, vocab_size, (B, S))

    # Forward pass
    logits, cache = model.forward(token_ids)

    # Backward pass
    loss, grads = model.backward(targets, cache)

    # Check loss is reasonable
    assert not np.isnan(loss), "Loss is NaN"
    assert not np.isinf(loss), "Loss is Inf"
    assert loss > 0, "Loss should be positive"

    # Check gradients exist for all parameters
    for param_name, param in model.params.items():
        if param_name == 'pos_encoding':  # Skip fixed positional encoding
            continue
        assert param_name in grads, f"Missing gradient for {param_name}"
        assert grads[param_name].shape == param.shape, f"Shape mismatch for {param_name}"
        assert not np.isnan(grads[param_name]).any(), f"Gradient for {param_name} contains NaN"
        assert not np.isinf(grads[param_name]).any(), f"Gradient for {param_name} contains Inf"

    print(f"  Loss: {loss:.4f}")
    print(f"  Number of gradients: {len(grads)}")
    print(f"  ✅ Backward pass works!\n")


def test_model_with_learned_positional():
    """Test model with learned positional embeddings."""
    print("Testing model with learned positional embeddings...")

    model = TransformerLM(
        vocab_size=50,
        d_model=32,
        n_heads=2,
        d_ff=64,
        n_layers=1,
        use_learned_pos=True,
        tie_embeddings=False
    )

    B, S = 2, 8
    token_ids = np.random.randint(0, 50, (B, S))
    targets = np.random.randint(0, 50, (B, S))

    logits, cache = model.forward(token_ids)
    loss, grads = model.backward(targets, cache)

    # Should have gradients for learned positional embeddings
    assert 'pos_embed' in grads, "Missing gradient for learned positional embeddings"
    assert 'lm_head_W' in grads, "Should have separate LM head when not tied"

    print(f"  Loss: {loss:.4f}")
    print(f"  ✅ Learned positional embeddings work!\n")


def test_model_with_padding():
    """Test model with padding tokens."""
    print("Testing model with padding tokens...")

    vocab_size = 50
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=2,
        d_ff=64,
        n_layers=1
    )

    B, S = 2, 8
    token_ids = np.random.randint(0, vocab_size, (B, S))
    targets = np.random.randint(0, vocab_size, (B, S))

    # Add padding to targets
    ignore_index = -1
    targets[:, -2:] = ignore_index  # Last 2 positions are padding

    logits, cache = model.forward(token_ids)
    loss, grads = model.backward(targets, cache, ignore_index=ignore_index)

    # Loss should ignore padded positions
    assert not np.isnan(loss), "Loss is NaN"
    assert loss > 0, "Loss should be positive"

    print(f"  Loss with padding: {loss:.4f}")
    print(f"  ✅ Padding handling works!\n")


def test_parameter_count():
    """Test parameter counting."""
    print("Testing parameter count...")

    model = TransformerLM(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        d_ff=512,
        n_layers=2,
        tie_embeddings=True
    )

    num_params = model.get_num_params()

    # Rough estimate:
    # Token embeddings: 1000 * 128 = 128k
    # Per layer (~4 * 128^2 QKV + 128^2 O + 128*512 + 512*128 FFN + 2*2*128 LN): ~300k
    # 2 layers: ~600k
    # Total: ~730k (with tied embeddings, no separate LM head)

    print(f"  Total parameters: {num_params:,}")
    assert 500_000 < num_params < 1_000_000, f"Unexpected parameter count: {num_params}"

    print(f"  ✅ Parameter count reasonable!\n")


def test_small_training_step():
    """Test a complete training step with gradient update."""
    print("Testing complete training step...")

    model = TransformerLM(
        vocab_size=50,
        d_model=32,
        n_heads=2,
        d_ff=64,
        n_layers=1
    )

    B, S = 2, 8
    token_ids = np.random.randint(0, 50, (B, S))
    targets = np.random.randint(0, 50, (B, S))

    # Forward
    logits1, cache = model.forward(token_ids)
    loss1, grads = model.backward(targets, cache)

    # Manual gradient update (simple SGD)
    lr = 0.01
    for param_name, param in model.params.items():
        if param_name == 'pos_encoding':  # Skip fixed encoding
            continue
        if param_name in grads:
            param -= lr * grads[param_name]

    # Forward again after update
    logits2, cache = model.forward(token_ids)
    loss2, _ = model.backward(targets, cache)

    # Logits should have changed
    logits_change = np.abs(logits2 - logits1).max()
    print(f"  Loss before: {loss1:.4f}")
    print(f"  Loss after: {loss2:.4f}")
    print(f"  Max logits change: {logits_change:.4f}")

    assert logits_change > 1e-4, "Parameters should have changed after update"
    print(f"  ✅ Training step works!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Complete Language Model")
    print("=" * 60)
    print()

    test_model_forward()
    test_model_backward()
    test_model_with_learned_positional()
    test_model_with_padding()
    test_parameter_count()
    test_small_training_step()

    print("=" * 60)
    print("🎉 All language model tests passed!")
    print("=" * 60)