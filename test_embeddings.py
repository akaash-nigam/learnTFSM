"""
Test token embeddings and positional encodings against PyTorch.
"""
import numpy as np
import torch
import torch.nn as nn
from embeddings import (
    token_embedding_forward, token_embedding_backward,
    sinusoidal_positional_encoding,
    learned_positional_embedding_forward, learned_positional_embedding_backward,
    embedding_layer_forward, embedding_layer_backward
)

np.random.seed(42)
torch.manual_seed(42)


def test_token_embedding():
    """Test token embedding lookup and backward pass."""
    print("Testing token embedding...")

    B, S = 4, 8
    vocab_size, d_model = 100, 64

    # Create token IDs
    token_ids = np.random.randint(0, vocab_size, (B, S))

    # Create embedding matrix
    embed_matrix = np.random.randn(vocab_size, d_model).astype(np.float32)

    # Forward pass - NumPy
    out_np = token_embedding_forward(token_ids, embed_matrix)

    # Forward pass - PyTorch
    embed_matrix_t = torch.from_numpy(embed_matrix).requires_grad_(True)
    token_ids_t = torch.from_numpy(token_ids).long()
    out_t = nn.functional.embedding(token_ids_t, embed_matrix_t)

    # Compare forward
    max_diff = np.abs(out_np - out_t.detach().numpy()).max()
    print(f"  Forward max diff: {max_diff:.2e}")
    assert max_diff < 1e-6, "Forward pass mismatch"

    # Backward pass
    dOut = np.random.randn(*out_np.shape).astype(np.float32)
    dEmbed_np = token_embedding_backward(token_ids, embed_matrix, dOut)

    # PyTorch backward
    dOut_t = torch.from_numpy(dOut)
    out_t.backward(dOut_t)

    # Compare gradients
    grad_diff = np.abs(dEmbed_np - embed_matrix_t.grad.numpy()).max()
    print(f"  Gradient max diff: {grad_diff:.2e}")
    assert grad_diff < 1e-6, "Gradient mismatch"

    print("  ✅ Token embedding passed!\n")


def test_sinusoidal_positional_encoding():
    """Test sinusoidal positional encoding generation."""
    print("Testing sinusoidal positional encoding...")

    seq_len, d_model = 20, 64

    # Generate encoding
    pos_enc = sinusoidal_positional_encoding(seq_len, d_model)

    # Basic sanity checks
    assert pos_enc.shape == (seq_len, d_model), "Wrong shape"
    assert not np.isnan(pos_enc).any(), "Contains NaN"
    assert not np.isinf(pos_enc).any(), "Contains Inf"

    # Check values are in reasonable range (sin/cos → [-1, 1])
    assert pos_enc.min() >= -1.0 and pos_enc.max() <= 1.0, "Values out of range"

    # Check first position is all zeros at even indices and ones at odd (approximately)
    # Actually, first position should be sin(0) = 0 and cos(0) = 1
    assert np.abs(pos_enc[0, 0]) < 1e-6, "First even position should be ~0"
    assert np.abs(pos_enc[0, 1] - 1.0) < 1e-6, "First odd position should be ~1"

    # Verify against PyTorch implementation
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))

    pos_enc_t = torch.zeros(seq_len, d_model)
    pos_enc_t[:, 0::2] = torch.sin(position * div_term)
    pos_enc_t[:, 1::2] = torch.cos(position * div_term)

    max_diff = np.abs(pos_enc - pos_enc_t.numpy()).max()
    print(f"  Max diff vs PyTorch: {max_diff:.2e}")
    assert max_diff < 1e-6, "Mismatch with PyTorch"

    print("  ✅ Sinusoidal positional encoding passed!\n")


def test_learned_positional_embedding():
    """Test learned positional embedding lookup and backward pass."""
    print("Testing learned positional embedding...")

    B, S = 4, 8
    max_seq_len, d_model = 512, 64

    # Create position indices
    positions = np.arange(S)[np.newaxis, :].repeat(B, axis=0)

    # Create position embedding matrix
    pos_embed_matrix = np.random.randn(max_seq_len, d_model).astype(np.float32)

    # Forward pass - NumPy
    out_np = learned_positional_embedding_forward(positions, pos_embed_matrix)

    # Forward pass - PyTorch
    pos_embed_matrix_t = torch.from_numpy(pos_embed_matrix).requires_grad_(True)
    positions_t = torch.from_numpy(positions).long()
    out_t = nn.functional.embedding(positions_t, pos_embed_matrix_t)

    # Compare forward
    max_diff = np.abs(out_np - out_t.detach().numpy()).max()
    print(f"  Forward max diff: {max_diff:.2e}")
    assert max_diff < 1e-6, "Forward pass mismatch"

    # Backward pass
    dOut = np.random.randn(*out_np.shape).astype(np.float32)
    dPosEmbed_np = learned_positional_embedding_backward(positions, pos_embed_matrix, dOut)

    # PyTorch backward
    dOut_t = torch.from_numpy(dOut)
    out_t.backward(dOut_t)

    # Compare gradients
    grad_diff = np.abs(dPosEmbed_np - pos_embed_matrix_t.grad.numpy()).max()
    print(f"  Gradient max diff: {grad_diff:.2e}")
    assert grad_diff < 1e-6, "Gradient mismatch"

    print("  ✅ Learned positional embedding passed!\n")


def test_embedding_layer_sinusoidal():
    """Test complete embedding layer with sinusoidal positional encoding."""
    print("Testing embedding layer (sinusoidal)...")

    B, S = 4, 8
    vocab_size, d_model = 100, 64
    max_seq_len = 512

    # Create inputs
    token_ids = np.random.randint(0, vocab_size, (B, S))
    token_embed_matrix = np.random.randn(vocab_size, d_model).astype(np.float32)
    pos_encoding = sinusoidal_positional_encoding(max_seq_len, d_model)

    # Forward pass - NumPy
    out_np = embedding_layer_forward(token_ids, token_embed_matrix, pos_encoding)

    # Forward pass - PyTorch
    token_embed_matrix_t = torch.from_numpy(token_embed_matrix).requires_grad_(True)
    pos_encoding_t = torch.from_numpy(pos_encoding)
    token_ids_t = torch.from_numpy(token_ids).long()

    token_embeds_t = nn.functional.embedding(token_ids_t, token_embed_matrix_t)
    positions_t = torch.arange(S).unsqueeze(0).repeat(B, 1)
    pos_embeds_t = pos_encoding_t[positions_t]
    out_t = token_embeds_t + pos_embeds_t

    # Compare forward
    max_diff = np.abs(out_np - out_t.detach().numpy()).max()
    print(f"  Forward max diff: {max_diff:.2e}")
    assert max_diff < 1e-6, "Forward pass mismatch"

    # Backward pass
    dOut = np.random.randn(*out_np.shape).astype(np.float32)
    dTokenEmbed_np, dPosEmbed_np = embedding_layer_backward(
        token_ids, token_embed_matrix, pos_encoding, dOut, use_learned_pos=False
    )

    # PyTorch backward
    dOut_t = torch.from_numpy(dOut)
    out_t.backward(dOut_t)

    # Compare gradients (only token embedding, pos is fixed)
    grad_diff = np.abs(dTokenEmbed_np - token_embed_matrix_t.grad.numpy()).max()
    print(f"  Token embedding gradient max diff: {grad_diff:.2e}")
    assert grad_diff < 1e-6, "Token embedding gradient mismatch"
    assert dPosEmbed_np is None, "Positional encoding should not have gradient"

    print("  ✅ Embedding layer (sinusoidal) passed!\n")


def test_embedding_layer_learned():
    """Test complete embedding layer with learned positional embeddings."""
    print("Testing embedding layer (learned)...")

    B, S = 4, 8
    vocab_size, d_model = 100, 64
    max_seq_len = 512

    # Create inputs
    token_ids = np.random.randint(0, vocab_size, (B, S))
    token_embed_matrix = np.random.randn(vocab_size, d_model).astype(np.float32)
    pos_embed_matrix = np.random.randn(max_seq_len, d_model).astype(np.float32)

    # Forward pass - NumPy
    out_np = embedding_layer_forward(token_ids, token_embed_matrix, pos_embed_matrix,
                                     use_learned_pos=True)

    # Forward pass - PyTorch
    token_embed_matrix_t = torch.from_numpy(token_embed_matrix).requires_grad_(True)
    pos_embed_matrix_t = torch.from_numpy(pos_embed_matrix).requires_grad_(True)
    token_ids_t = torch.from_numpy(token_ids).long()

    token_embeds_t = nn.functional.embedding(token_ids_t, token_embed_matrix_t)
    positions_t = torch.arange(S).unsqueeze(0).repeat(B, 1)
    pos_embeds_t = nn.functional.embedding(positions_t, pos_embed_matrix_t)
    out_t = token_embeds_t + pos_embeds_t

    # Compare forward
    max_diff = np.abs(out_np - out_t.detach().numpy()).max()
    print(f"  Forward max diff: {max_diff:.2e}")
    assert max_diff < 1e-6, "Forward pass mismatch"

    # Backward pass
    dOut = np.random.randn(*out_np.shape).astype(np.float32)
    dTokenEmbed_np, dPosEmbed_np = embedding_layer_backward(
        token_ids, token_embed_matrix, pos_embed_matrix, dOut, use_learned_pos=True
    )

    # PyTorch backward
    dOut_t = torch.from_numpy(dOut)
    out_t.backward(dOut_t)

    # Compare gradients
    token_grad_diff = np.abs(dTokenEmbed_np - token_embed_matrix_t.grad.numpy()).max()
    pos_grad_diff = np.abs(dPosEmbed_np - pos_embed_matrix_t.grad.numpy()).max()

    print(f"  Token embedding gradient max diff: {token_grad_diff:.2e}")
    print(f"  Positional embedding gradient max diff: {pos_grad_diff:.2e}")

    assert token_grad_diff < 1e-6, "Token embedding gradient mismatch"
    assert pos_grad_diff < 1e-6, "Positional embedding gradient mismatch"

    print("  ✅ Embedding layer (learned) passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Embedding Layers")
    print("=" * 60)
    print()

    test_token_embedding()
    test_sinusoidal_positional_encoding()
    test_learned_positional_embedding()
    test_embedding_layer_sinusoidal()
    test_embedding_layer_learned()

    print("=" * 60)
    print("🎉 All embedding tests passed!")
    print("=" * 60)