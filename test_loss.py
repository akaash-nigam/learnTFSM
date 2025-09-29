"""
Test cross-entropy loss implementation against PyTorch.
"""
import numpy as np
import torch
import torch.nn.functional as F
from loss import (
    cross_entropy_loss,
    cross_entropy_loss_backward,
    cross_entropy_loss_with_logits,
    perplexity,
    accuracy
)

np.random.seed(42)
torch.manual_seed(42)


def test_cross_entropy_loss_forward():
    """Test cross-entropy loss computation."""
    print("Testing cross-entropy loss forward...")

    B, S, vocab_size = 4, 8, 100

    # Create logits and targets
    logits = np.random.randn(B, S, vocab_size).astype(np.float32)
    targets = np.random.randint(0, vocab_size, (B, S))

    # NumPy version
    loss_np = cross_entropy_loss(logits, targets)

    # PyTorch version
    logits_t = torch.from_numpy(logits).requires_grad_(True)
    targets_t = torch.from_numpy(targets).long()

    logits_flat = logits_t.reshape(-1, vocab_size)
    targets_flat = targets_t.reshape(-1)
    loss_t = F.cross_entropy(logits_flat, targets_flat)

    # Compare
    diff = abs(loss_np - loss_t.item())
    print(f"  Loss difference: {diff:.2e}")
    print(f"  NumPy loss: {loss_np:.6f}")
    print(f"  PyTorch loss: {loss_t.item():.6f}")
    assert diff < 1e-6, f"Loss mismatch: {diff}"

    print("  ✅ Cross-entropy loss forward passed!\n")


def test_cross_entropy_loss_backward():
    """Test cross-entropy loss gradient."""
    print("Testing cross-entropy loss backward...")

    B, S, vocab_size = 4, 8, 100

    # Create logits and targets
    logits = np.random.randn(B, S, vocab_size).astype(np.float32)
    targets = np.random.randint(0, vocab_size, (B, S))

    # NumPy backward
    dLogits_np = cross_entropy_loss_backward(logits, targets)

    # PyTorch backward
    logits_t = torch.from_numpy(logits).requires_grad_(True)
    targets_t = torch.from_numpy(targets).long()

    logits_flat = logits_t.reshape(-1, vocab_size)
    targets_flat = targets_t.reshape(-1)
    loss_t = F.cross_entropy(logits_flat, targets_flat)
    loss_t.backward()

    dLogits_t = logits_t.grad.numpy()

    # Compare
    max_diff = np.abs(dLogits_np - dLogits_t).max()
    rel_error = max_diff / (np.abs(dLogits_t).max() + 1e-8)

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Relative error: {rel_error:.2e}")
    assert max_diff < 1e-6, f"Gradient mismatch: {max_diff}"

    print("  ✅ Cross-entropy loss backward passed!\n")


def test_cross_entropy_with_ignore_index():
    """Test cross-entropy with padding/ignored tokens."""
    print("Testing cross-entropy with ignore_index...")

    B, S, vocab_size = 4, 8, 100
    ignore_index = -1

    # Create logits and targets with some padding
    logits = np.random.randn(B, S, vocab_size).astype(np.float32)
    targets = np.random.randint(0, vocab_size, (B, S))

    # Set some positions to padding
    targets[:, -2:] = ignore_index  # Last 2 positions are padding

    # NumPy version
    loss_np = cross_entropy_loss(logits, targets, ignore_index=ignore_index)
    dLogits_np = cross_entropy_loss_backward(logits, targets, ignore_index=ignore_index)

    # PyTorch version
    logits_t = torch.from_numpy(logits).requires_grad_(True)
    targets_t = torch.from_numpy(targets).long()

    logits_flat = logits_t.reshape(-1, vocab_size)
    targets_flat = targets_t.reshape(-1)
    loss_t = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)
    loss_t.backward()

    # Compare loss
    loss_diff = abs(loss_np - loss_t.item())
    print(f"  Loss difference: {loss_diff:.2e}")
    assert loss_diff < 1e-6, f"Loss mismatch with ignore_index: {loss_diff}"

    # Compare gradient
    dLogits_t = logits_t.grad.numpy()
    max_diff = np.abs(dLogits_np - dLogits_t).max()
    print(f"  Gradient max difference: {max_diff:.2e}")
    assert max_diff < 1e-6, f"Gradient mismatch with ignore_index: {max_diff}"

    # Verify padded positions have zero gradient
    padded_grad = dLogits_np[:, -2:, :]
    assert np.abs(padded_grad).max() < 1e-10, "Padded positions should have zero gradient"
    print(f"  Padded gradient max: {np.abs(padded_grad).max():.2e}")

    print("  ✅ Cross-entropy with ignore_index passed!\n")


def test_cross_entropy_combined():
    """Test combined loss and gradient computation."""
    print("Testing cross_entropy_loss_with_logits...")

    B, S, vocab_size = 4, 8, 100

    logits = np.random.randn(B, S, vocab_size).astype(np.float32)
    targets = np.random.randint(0, vocab_size, (B, S))

    # Combined version
    loss_combined, dLogits_combined = cross_entropy_loss_with_logits(logits, targets)

    # Separate versions
    loss_separate = cross_entropy_loss(logits, targets)
    dLogits_separate = cross_entropy_loss_backward(logits, targets)

    # Compare
    loss_diff = abs(loss_combined - loss_separate)
    grad_diff = np.abs(dLogits_combined - dLogits_separate).max()

    print(f"  Loss difference: {loss_diff:.2e}")
    print(f"  Gradient difference: {grad_diff:.2e}")

    assert loss_diff < 1e-10, "Combined loss differs from separate"
    assert grad_diff < 1e-10, "Combined gradient differs from separate"

    print("  ✅ Combined loss/gradient passed!\n")


def test_perplexity():
    """Test perplexity computation."""
    print("Testing perplexity...")

    B, S, vocab_size = 4, 8, 100

    logits = np.random.randn(B, S, vocab_size).astype(np.float32)
    targets = np.random.randint(0, vocab_size, (B, S))

    # Compute perplexity
    ppl_np = perplexity(logits, targets)
    loss_np = cross_entropy_loss(logits, targets)

    # Verify ppl = exp(loss)
    ppl_expected = np.exp(loss_np)
    diff = abs(ppl_np - ppl_expected)

    print(f"  Perplexity: {ppl_np:.4f}")
    print(f"  Expected (exp(loss)): {ppl_expected:.4f}")
    print(f"  Difference: {diff:.2e}")

    assert diff < 1e-6, "Perplexity mismatch"

    print("  ✅ Perplexity passed!\n")


def test_accuracy():
    """Test accuracy computation."""
    print("Testing accuracy...")

    B, S, vocab_size = 4, 8, 100

    # Create logits where some predictions are correct
    logits = np.random.randn(B, S, vocab_size).astype(np.float32)
    targets = np.random.randint(0, vocab_size, (B, S))

    # Make first half correct by setting max logit to target
    for b in range(B):
        for s in range(S // 2):
            logits[b, s, :] = -10.0
            logits[b, s, targets[b, s]] = 10.0

    # Compute accuracy
    acc = accuracy(logits, targets)

    # Should be around 50% (first half correct)
    print(f"  Accuracy: {acc:.4f}")
    assert 0.45 <= acc <= 0.55, f"Expected ~0.5, got {acc}"

    # Test with ignore_index
    targets_with_padding = targets.copy()
    targets_with_padding[:, -2:] = -1  # Pad last 2

    acc_with_ignore = accuracy(logits, targets_with_padding, ignore_index=-1)
    print(f"  Accuracy (with padding): {acc_with_ignore:.4f}")

    # Should be higher since we padded some incorrect predictions
    expected_valid = S - 2
    expected_correct = S // 2  # Still have first half correct
    expected_acc = expected_correct / expected_valid
    print(f"  Expected accuracy: {expected_acc:.4f}")

    assert abs(acc_with_ignore - expected_acc) < 0.05, "Accuracy with padding incorrect"

    print("  ✅ Accuracy passed!\n")


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    print("Testing numerical stability...")

    B, S, vocab_size = 2, 4, 50

    # Test with large positive values
    logits_large = np.random.randn(B, S, vocab_size).astype(np.float32) * 100
    targets = np.random.randint(0, vocab_size, (B, S))

    loss_large = cross_entropy_loss(logits_large, targets)
    dLogits_large = cross_entropy_loss_backward(logits_large, targets)

    assert not np.isnan(loss_large), "Loss is NaN with large values"
    assert not np.isnan(dLogits_large).any(), "Gradients contain NaN"
    assert not np.isinf(loss_large), "Loss is Inf with large values"
    assert not np.isinf(dLogits_large).any(), "Gradients contain Inf"

    print(f"  Loss with large values: {loss_large:.4f}")
    print(f"  Gradient range: [{dLogits_large.min():.2e}, {dLogits_large.max():.2e}]")

    print("  ✅ Numerical stability passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Cross-Entropy Loss")
    print("=" * 60)
    print()

    test_cross_entropy_loss_forward()
    test_cross_entropy_loss_backward()
    test_cross_entropy_with_ignore_index()
    test_cross_entropy_combined()
    test_perplexity()
    test_accuracy()
    test_numerical_stability()

    print("=" * 60)
    print("🎉 All loss function tests passed!")
    print("=" * 60)