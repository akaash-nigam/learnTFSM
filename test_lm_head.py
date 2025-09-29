"""
Test language model head implementation against PyTorch.
"""
import numpy as np
import torch
import torch.nn as nn
from lm_head import (
    lm_head_forward,
    lm_head_backward,
    tied_lm_head_forward,
    tied_lm_head_backward
)

np.random.seed(42)
torch.manual_seed(42)


def test_lm_head_forward():
    """Test language model head forward pass."""
    print("Testing LM head forward...")

    B, S, d_model = 4, 8, 64
    vocab_size = 100

    # Create inputs
    hidden_states = np.random.randn(B, S, d_model).astype(np.float32)
    W_lm = np.random.randn(d_model, vocab_size).astype(np.float32)
    b_lm = np.random.randn(vocab_size).astype(np.float32)

    # NumPy forward
    logits_np = lm_head_forward(hidden_states, W_lm, b_lm)

    # PyTorch forward
    hidden_states_t = torch.from_numpy(hidden_states).requires_grad_(True)
    W_lm_t = torch.from_numpy(W_lm).requires_grad_(True)
    b_lm_t = torch.from_numpy(b_lm).requires_grad_(True)

    hidden_2d_t = hidden_states_t.reshape(-1, d_model)
    logits_2d_t = torch.matmul(hidden_2d_t, W_lm_t) + b_lm_t
    logits_t = logits_2d_t.reshape(B, S, vocab_size)

    # Compare
    max_diff = np.abs(logits_np - logits_t.detach().numpy()).max()
    print(f"  Max difference: {max_diff:.2e}")
    assert max_diff < 1e-6, "Forward pass mismatch"

    print("  ✅ LM head forward passed!\n")


def test_lm_head_backward():
    """Test language model head backward pass."""
    print("Testing LM head backward...")

    B, S, d_model = 4, 8, 64
    vocab_size = 100

    # Create inputs
    hidden_states = np.random.randn(B, S, d_model).astype(np.float32)
    W_lm = np.random.randn(d_model, vocab_size).astype(np.float32)
    b_lm = np.random.randn(vocab_size).astype(np.float32)

    # Forward
    logits_np = lm_head_forward(hidden_states, W_lm, b_lm)

    # Backward - NumPy
    dLogits = np.random.randn(*logits_np.shape).astype(np.float32)
    dHidden_np, dW_lm_np, dB_lm_np = lm_head_backward(hidden_states, W_lm, dLogits, b_lm)

    # PyTorch
    hidden_states_t = torch.from_numpy(hidden_states).requires_grad_(True)
    W_lm_t = torch.from_numpy(W_lm).requires_grad_(True)
    b_lm_t = torch.from_numpy(b_lm).requires_grad_(True)

    hidden_2d_t = hidden_states_t.reshape(-1, d_model)
    logits_2d_t = torch.matmul(hidden_2d_t, W_lm_t) + b_lm_t
    logits_t = logits_2d_t.reshape(B, S, vocab_size)

    dLogits_t = torch.from_numpy(dLogits)
    logits_t.backward(dLogits_t)

    # Compare gradients
    dHidden_diff = np.abs(dHidden_np - hidden_states_t.grad.numpy()).max()
    dW_diff = np.abs(dW_lm_np - W_lm_t.grad.numpy()).max()
    dB_diff = np.abs(dB_lm_np - b_lm_t.grad.numpy()).max()

    print(f"  dHidden max diff: {dHidden_diff:.2e}")
    print(f"  dW max diff: {dW_diff:.2e}")
    print(f"  dB max diff: {dB_diff:.2e}")

    assert dHidden_diff < 1e-5, "Hidden gradient mismatch"
    assert dW_diff < 1e-5, "Weight gradient mismatch"
    assert dB_diff < 1e-5, "Bias gradient mismatch"

    print("  ✅ LM head backward passed!\n")


def test_lm_head_no_bias():
    """Test language model head without bias."""
    print("Testing LM head without bias...")

    B, S, d_model = 4, 8, 64
    vocab_size = 100

    # Create inputs
    hidden_states = np.random.randn(B, S, d_model).astype(np.float32)
    W_lm = np.random.randn(d_model, vocab_size).astype(np.float32)

    # Forward
    logits_np = lm_head_forward(hidden_states, W_lm, b_lm=None)

    # Backward
    dLogits = np.random.randn(*logits_np.shape).astype(np.float32)
    dHidden_np, dW_lm_np, dB_lm_np = lm_head_backward(hidden_states, W_lm, dLogits, b_lm=None)

    # PyTorch
    hidden_states_t = torch.from_numpy(hidden_states).requires_grad_(True)
    W_lm_t = torch.from_numpy(W_lm).requires_grad_(True)

    hidden_2d_t = hidden_states_t.reshape(-1, d_model)
    logits_2d_t = torch.matmul(hidden_2d_t, W_lm_t)
    logits_t = logits_2d_t.reshape(B, S, vocab_size)

    dLogits_t = torch.from_numpy(dLogits)
    logits_t.backward(dLogits_t)

    # Compare
    logits_diff = np.abs(logits_np - logits_t.detach().numpy()).max()
    dHidden_diff = np.abs(dHidden_np - hidden_states_t.grad.numpy()).max()
    dW_diff = np.abs(dW_lm_np - W_lm_t.grad.numpy()).max()

    print(f"  Logits max diff: {logits_diff:.2e}")
    print(f"  dHidden max diff: {dHidden_diff:.2e}")
    print(f"  dW max diff: {dW_diff:.2e}")

    assert logits_diff < 1e-6, "Forward mismatch"
    assert dHidden_diff < 1e-6, "Hidden gradient mismatch"
    assert dW_diff < 1e-6, "Weight gradient mismatch"
    assert dB_lm_np is None, "Bias gradient should be None"

    print("  ✅ LM head without bias passed!\n")


def test_tied_lm_head_forward():
    """Test tied LM head (weight sharing with embeddings)."""
    print("Testing tied LM head forward...")

    B, S, d_model = 4, 8, 64
    vocab_size = 100

    # Create inputs
    hidden_states = np.random.randn(B, S, d_model).astype(np.float32)
    token_embed_matrix = np.random.randn(vocab_size, d_model).astype(np.float32)

    # NumPy forward
    logits_np = tied_lm_head_forward(hidden_states, token_embed_matrix)

    # PyTorch forward
    hidden_states_t = torch.from_numpy(hidden_states).requires_grad_(True)
    token_embed_matrix_t = torch.from_numpy(token_embed_matrix).requires_grad_(True)

    hidden_2d_t = hidden_states_t.reshape(-1, d_model)
    logits_2d_t = torch.matmul(hidden_2d_t, token_embed_matrix_t.T)
    logits_t = logits_2d_t.reshape(B, S, vocab_size)

    # Compare
    max_diff = np.abs(logits_np - logits_t.detach().numpy()).max()
    print(f"  Max difference: {max_diff:.2e}")
    assert max_diff < 1e-6, "Forward pass mismatch"

    print("  ✅ Tied LM head forward passed!\n")


def test_tied_lm_head_backward():
    """Test tied LM head backward pass."""
    print("Testing tied LM head backward...")

    B, S, d_model = 4, 8, 64
    vocab_size = 100

    # Create inputs
    hidden_states = np.random.randn(B, S, d_model).astype(np.float32)
    token_embed_matrix = np.random.randn(vocab_size, d_model).astype(np.float32)

    # Forward
    logits_np = tied_lm_head_forward(hidden_states, token_embed_matrix)

    # Backward - NumPy
    dLogits = np.random.randn(*logits_np.shape).astype(np.float32)
    dHidden_np, dEmbed_np = tied_lm_head_backward(hidden_states, token_embed_matrix, dLogits)

    # PyTorch
    hidden_states_t = torch.from_numpy(hidden_states).requires_grad_(True)
    token_embed_matrix_t = torch.from_numpy(token_embed_matrix).requires_grad_(True)

    hidden_2d_t = hidden_states_t.reshape(-1, d_model)
    logits_2d_t = torch.matmul(hidden_2d_t, token_embed_matrix_t.T)
    logits_t = logits_2d_t.reshape(B, S, vocab_size)

    dLogits_t = torch.from_numpy(dLogits)
    logits_t.backward(dLogits_t)

    # Compare gradients
    dHidden_diff = np.abs(dHidden_np - hidden_states_t.grad.numpy()).max()
    dEmbed_diff = np.abs(dEmbed_np - token_embed_matrix_t.grad.numpy()).max()

    print(f"  dHidden max diff: {dHidden_diff:.2e}")
    print(f"  dEmbed max diff: {dEmbed_diff:.2e}")

    assert dHidden_diff < 1e-6, "Hidden gradient mismatch"
    assert dEmbed_diff < 1e-6, "Embedding gradient mismatch"

    print("  ✅ Tied LM head backward passed!\n")


def test_tied_vs_untied():
    """Compare tied and untied LM heads to verify they're equivalent when W_lm = embed^T."""
    print("Testing tied vs untied equivalence...")

    B, S, d_model = 4, 8, 64
    vocab_size = 100

    # Create inputs
    hidden_states = np.random.randn(B, S, d_model).astype(np.float32)
    token_embed_matrix = np.random.randn(vocab_size, d_model).astype(np.float32)

    # Untied: use transpose of embedding as weight
    W_lm = token_embed_matrix.T  # (d_model, vocab_size)

    # Forward passes
    logits_untied = lm_head_forward(hidden_states, W_lm, b_lm=None)
    logits_tied = tied_lm_head_forward(hidden_states, token_embed_matrix)

    # Should be identical
    forward_diff = np.abs(logits_untied - logits_tied).max()
    print(f"  Forward difference: {forward_diff:.2e}")
    assert forward_diff < 1e-10, "Tied and untied forward should match"

    # Backward passes
    dLogits = np.random.randn(*logits_tied.shape).astype(np.float32)

    dHidden_untied, dW_lm, _ = lm_head_backward(hidden_states, W_lm, dLogits, b_lm=None)
    dHidden_tied, dEmbed = tied_lm_head_backward(hidden_states, token_embed_matrix, dLogits)

    # Hidden gradients should match
    hidden_diff = np.abs(dHidden_untied - dHidden_tied).max()
    print(f"  Hidden gradient difference: {hidden_diff:.2e}")
    assert hidden_diff < 1e-10, "Hidden gradients should match"

    # Embedding gradient should be transpose of weight gradient
    embed_diff = np.abs(dEmbed - dW_lm.T).max()
    print(f"  Embedding gradient difference: {embed_diff:.2e}")
    assert embed_diff < 1e-10, "Embedding grad should equal W_lm.T grad"

    print("  ✅ Tied vs untied equivalence passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Language Model Head")
    print("=" * 60)
    print()

    test_lm_head_forward()
    test_lm_head_backward()
    test_lm_head_no_bias()
    test_tied_lm_head_forward()
    test_tied_lm_head_backward()
    test_tied_vs_untied()

    print("=" * 60)
    print("🎉 All LM head tests passed!")
    print("=" * 60)