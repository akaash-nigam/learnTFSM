"""
Test causal masking implementation in multi-head attention.
"""
import numpy as np
import torch
import torch.nn.functional as F
from transformer import multi_head_attention, multi_head_attention_backward

np.random.seed(42)
torch.manual_seed(42)

def test_causal_masking_forward():
    """Test that causal masking prevents attending to future positions."""
    print("Testing causal masking in forward pass...")

    # Simple test case
    B, H, S, D = 1, 1, 4, 8

    # Create simple inputs
    q = np.random.randn(B, H, S, D).astype(np.float32)
    k = np.random.randn(B, H, S, D).astype(np.float32)
    v = np.random.randn(B, H, S, D).astype(np.float32)

    # Forward pass with causal masking
    out_causal = multi_head_attention(q, k, v, causal=True)

    # Forward pass without causal masking
    out_no_causal = multi_head_attention(q, k, v, causal=False)

    # They should be different
    assert not np.allclose(out_causal, out_no_causal), "Causal and non-causal outputs should differ"

    # Verify with PyTorch
    q_t = torch.from_numpy(q).requires_grad_(True)
    k_t = torch.from_numpy(k).requires_grad_(True)
    v_t = torch.from_numpy(v).requires_grad_(True)

    scale = 1.0 / np.sqrt(D)
    scores_t = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

    # Apply causal mask
    causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
    scores_t = scores_t.masked_fill(causal_mask, float('-inf'))

    attn_t = F.softmax(scores_t, dim=-1)
    out_t = torch.matmul(attn_t, v_t)

    # Compare
    max_diff = np.abs(out_causal - out_t.detach().numpy()).max()
    print(f"  Max difference between NumPy and PyTorch: {max_diff:.2e}")
    assert max_diff < 1e-6, f"Forward pass outputs don't match! Max diff: {max_diff}"
    print("  ✅ Causal masking forward pass correct!\n")

def test_causal_masking_backward():
    """Test that causal masking backward pass is correct."""
    print("Testing causal masking in backward pass...")

    B, H, S, D = 1, 2, 5, 8

    # Create inputs
    q = np.random.randn(B, H, S, D).astype(np.float32)
    k = np.random.randn(B, H, S, D).astype(np.float32)
    v = np.random.randn(B, H, S, D).astype(np.float32)

    # Forward pass
    out_np = multi_head_attention(q, k, v, causal=True)

    # Backward pass
    dOut = np.random.randn(*out_np.shape).astype(np.float32)
    dQ_np, dK_np, dV_np = multi_head_attention_backward(q, k, v, dOut, causal=True)

    # PyTorch version
    q_t = torch.from_numpy(q).requires_grad_(True)
    k_t = torch.from_numpy(k).requires_grad_(True)
    v_t = torch.from_numpy(v).requires_grad_(True)

    scale = 1.0 / np.sqrt(D)
    scores_t = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

    # Apply causal mask
    causal_mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
    scores_t = scores_t.masked_fill(causal_mask, float('-inf'))

    attn_t = F.softmax(scores_t, dim=-1)
    out_t = torch.matmul(attn_t, v_t)

    # Backward
    dOut_t = torch.from_numpy(dOut)
    out_t.backward(dOut_t)

    # Compare gradients
    dQ_diff = np.abs(dQ_np - q_t.grad.numpy()).max()
    dK_diff = np.abs(dK_np - k_t.grad.numpy()).max()
    dV_diff = np.abs(dV_np - v_t.grad.numpy()).max()

    print(f"  dQ max diff: {dQ_diff:.2e}")
    print(f"  dK max diff: {dK_diff:.2e}")
    print(f"  dV max diff: {dV_diff:.2e}")

    assert dQ_diff < 1e-5, f"dQ gradient mismatch: {dQ_diff}"
    assert dK_diff < 1e-5, f"dK gradient mismatch: {dK_diff}"
    assert dV_diff < 1e-5, f"dV gradient mismatch: {dV_diff}"

    print("  ✅ Causal masking backward pass correct!\n")

def test_causal_property():
    """Test that causal masking actually prevents information flow from future."""
    print("Testing causal property (no future information leakage)...")

    B, H, S, D = 1, 1, 4, 4

    # Create inputs where position i only depends on positions <= i
    q = np.random.randn(B, H, S, D).astype(np.float32)
    k = np.random.randn(B, H, S, D).astype(np.float32)
    v = np.random.randn(B, H, S, D).astype(np.float32)

    # Get output with causal masking
    out = multi_head_attention(q, k, v, causal=True)

    # Modify a future position in k and v
    k_modified = k.copy()
    v_modified = v.copy()
    k_modified[:, :, -1, :] += 100.0  # Huge change to last position
    v_modified[:, :, -1, :] += 100.0

    out_modified = multi_head_attention(q, k_modified, v_modified, causal=True)

    # First 3 positions should be unchanged (they can't see position 3)
    for i in range(S - 1):
        pos_diff = np.abs(out[:, :, i, :] - out_modified[:, :, i, :]).max()
        assert pos_diff < 1e-6, f"Position {i} was affected by future (diff: {pos_diff})"
        print(f"  Position {i}: ✅ (max diff: {pos_diff:.2e})")

    # Last position should be affected
    last_diff = np.abs(out[:, :, -1, :] - out_modified[:, :, -1, :]).max()
    assert last_diff > 1e-3, f"Last position should be affected (diff: {last_diff})"
    print(f"  Position {S-1}: ✅ affected as expected (max diff: {last_diff:.2e})")

    print("  ✅ Causal property verified!\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Causal Masking Implementation")
    print("=" * 60)
    print()

    test_causal_masking_forward()
    test_causal_masking_backward()
    test_causal_property()

    print("=" * 60)
    print("🎉 All causal masking tests passed!")
    print("=" * 60)