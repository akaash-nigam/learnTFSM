"""
Language model head for next-token prediction.
"""
import numpy as np
from typing import Tuple


def lm_head_forward(hidden_states: np.ndarray, W_lm: np.ndarray,
                    b_lm: np.ndarray = None) -> np.ndarray:
    """
    Project hidden states to vocabulary logits for next-token prediction.

    Args:
        hidden_states: (B, S, d_model) hidden states from transformer
        W_lm: (d_model, vocab_size) weight matrix
        b_lm: Optional (vocab_size,) bias vector

    Returns:
        logits: (B, S, vocab_size) unnormalized logits for each token
    """
    B, S, d_model = hidden_states.shape
    vocab_size = W_lm.shape[1]

    # Reshape to 2D for matrix multiplication
    hidden_2d = hidden_states.reshape(-1, d_model)  # (B*S, d_model)

    # Project to vocabulary
    logits_2d = hidden_2d @ W_lm  # (B*S, vocab_size)

    if b_lm is not None:
        logits_2d = logits_2d + b_lm

    # Reshape back to 3D
    logits = logits_2d.reshape(B, S, vocab_size)

    return logits


def lm_head_backward(hidden_states: np.ndarray, W_lm: np.ndarray,
                     dLogits: np.ndarray, b_lm: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass for language model head.

    Args:
        hidden_states: (B, S, d_model) hidden states from transformer
        W_lm: (d_model, vocab_size) weight matrix
        dLogits: (B, S, vocab_size) gradient from upstream
        b_lm: Optional (vocab_size,) bias vector

    Returns:
        dHidden: (B, S, d_model) gradient w.r.t. hidden states
        dW_lm: (d_model, vocab_size) gradient w.r.t. weight matrix
        dB_lm: (vocab_size,) gradient w.r.t. bias (None if no bias)
    """
    B, S, d_model = hidden_states.shape
    vocab_size = W_lm.shape[1]

    # Flatten
    hidden_2d = hidden_states.reshape(-1, d_model)  # (B*S, d_model)
    dLogits_2d = dLogits.reshape(-1, vocab_size)    # (B*S, vocab_size)

    # Gradients
    # dL/dW = hidden^T @ dLogits
    dW_lm = hidden_2d.T @ dLogits_2d  # (d_model, vocab_size)

    # dL/dHidden = dLogits @ W^T
    dHidden_2d = dLogits_2d @ W_lm.T  # (B*S, d_model)

    # dL/dB = sum over batch and sequence
    if b_lm is not None:
        dB_lm = dLogits_2d.sum(axis=0)  # (vocab_size,)
    else:
        dB_lm = None

    # Reshape hidden gradient back to 3D
    dHidden = dHidden_2d.reshape(B, S, d_model)

    return dHidden, dW_lm, dB_lm


def tied_lm_head_forward(hidden_states: np.ndarray,
                         token_embed_matrix: np.ndarray) -> np.ndarray:
    """
    Language model head with tied embeddings (weight sharing).
    Uses the transpose of token embedding matrix as the output projection.

    This is a common practice that:
    - Reduces parameters
    - Can improve performance
    - Makes intuitive sense (similar tokens have similar embeddings and predictions)

    Args:
        hidden_states: (B, S, d_model) hidden states from transformer
        token_embed_matrix: (vocab_size, d_model) token embedding matrix

    Returns:
        logits: (B, S, vocab_size) unnormalized logits
    """
    B, S, d_model = hidden_states.shape
    vocab_size = token_embed_matrix.shape[0]

    # Reshape and multiply by transpose of embedding matrix
    hidden_2d = hidden_states.reshape(-1, d_model)  # (B*S, d_model)
    logits_2d = hidden_2d @ token_embed_matrix.T    # (B*S, vocab_size)

    logits = logits_2d.reshape(B, S, vocab_size)
    return logits


def tied_lm_head_backward(hidden_states: np.ndarray,
                         token_embed_matrix: np.ndarray,
                         dLogits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward pass for tied language model head.

    Args:
        hidden_states: (B, S, d_model) hidden states
        token_embed_matrix: (vocab_size, d_model) token embedding matrix
        dLogits: (B, S, vocab_size) gradient from upstream

    Returns:
        dHidden: (B, S, d_model) gradient w.r.t. hidden states
        dEmbed: (vocab_size, d_model) gradient w.r.t. embedding matrix
    """
    B, S, d_model = hidden_states.shape
    vocab_size = token_embed_matrix.shape[0]

    # Flatten
    hidden_2d = hidden_states.reshape(-1, d_model)    # (B*S, d_model)
    dLogits_2d = dLogits.reshape(-1, vocab_size)      # (B*S, vocab_size)

    # Gradient w.r.t. hidden states
    # dL/dHidden = dLogits @ Embed (not transposed!)
    dHidden_2d = dLogits_2d @ token_embed_matrix  # (B*S, d_model)
    dHidden = dHidden_2d.reshape(B, S, d_model)

    # Gradient w.r.t. embedding matrix
    # dL/dEmbed = dLogits^T @ hidden
    dEmbed = dLogits_2d.T @ hidden_2d  # (vocab_size, d_model)

    return dHidden, dEmbed