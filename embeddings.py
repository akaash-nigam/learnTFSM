"""
Token embeddings and positional encodings for the Transformer language model.
"""
import numpy as np
from typing import Tuple


def token_embedding_forward(token_ids: np.ndarray, embed_matrix: np.ndarray) -> np.ndarray:
    """
    Look up embeddings for token IDs.

    Args:
        token_ids: (B, S) integer array of token indices
        embed_matrix: (vocab_size, d_model) embedding matrix

    Returns:
        embeddings: (B, S, d_model) embedded tokens
    """
    return embed_matrix[token_ids]


def token_embedding_backward(token_ids: np.ndarray, embed_matrix: np.ndarray,
                             dOut: np.ndarray) -> np.ndarray:
    """
    Backward pass for token embeddings.

    Args:
        token_ids: (B, S) integer array of token indices
        embed_matrix: (vocab_size, d_model) embedding matrix
        dOut: (B, S, d_model) gradient from upstream

    Returns:
        dEmbed: (vocab_size, d_model) gradient for embedding matrix
    """
    vocab_size, d_model = embed_matrix.shape
    dEmbed = np.zeros_like(embed_matrix)

    # Accumulate gradients for each token
    B, S = token_ids.shape
    for b in range(B):
        for s in range(S):
            token_id = token_ids[b, s]
            dEmbed[token_id] += dOut[b, s]

    return dEmbed


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings as in "Attention is All You Need".

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: Maximum sequence length
        d_model: Model dimension

    Returns:
        pos_encoding: (seq_len, d_model) positional encoding matrix
    """
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return pos_encoding


def learned_positional_embedding_forward(positions: np.ndarray,
                                        pos_embed_matrix: np.ndarray) -> np.ndarray:
    """
    Look up learned positional embeddings.

    Args:
        positions: (B, S) integer array of position indices (0 to max_seq_len-1)
        pos_embed_matrix: (max_seq_len, d_model) learned position embedding matrix

    Returns:
        pos_embeddings: (B, S, d_model) positional embeddings
    """
    return pos_embed_matrix[positions]


def learned_positional_embedding_backward(positions: np.ndarray,
                                          pos_embed_matrix: np.ndarray,
                                          dOut: np.ndarray) -> np.ndarray:
    """
    Backward pass for learned positional embeddings.

    Args:
        positions: (B, S) integer array of position indices
        pos_embed_matrix: (max_seq_len, d_model) learned position embedding matrix
        dOut: (B, S, d_model) gradient from upstream

    Returns:
        dPosEmbed: (max_seq_len, d_model) gradient for position embedding matrix
    """
    max_seq_len, d_model = pos_embed_matrix.shape
    dPosEmbed = np.zeros_like(pos_embed_matrix)

    # Accumulate gradients for each position
    B, S = positions.shape
    for b in range(B):
        for s in range(S):
            pos = positions[b, s]
            dPosEmbed[pos] += dOut[b, s]

    return dPosEmbed


def add_positional_encoding(token_embeds: np.ndarray,
                           pos_encoding: np.ndarray,
                           positions: np.ndarray = None) -> np.ndarray:
    """
    Add positional encoding to token embeddings.

    Args:
        token_embeds: (B, S, d_model) token embeddings
        pos_encoding: (max_seq_len, d_model) positional encoding/embedding matrix
        positions: Optional (B, S) position indices. If None, uses range(S).

    Returns:
        combined: (B, S, d_model) token embeddings + positional encoding
    """
    B, S, d_model = token_embeds.shape

    if positions is None:
        # Default to sequential positions [0, 1, 2, ..., S-1]
        positions = np.arange(S)[np.newaxis, :].repeat(B, axis=0)

    # Look up positional encodings for these positions
    pos_embeds = pos_encoding[positions]

    return token_embeds + pos_embeds


def embedding_layer_forward(token_ids: np.ndarray,
                            token_embed_matrix: np.ndarray,
                            pos_encoding: np.ndarray,
                            use_learned_pos: bool = False,
                            positions: np.ndarray = None) -> np.ndarray:
    """
    Complete embedding layer: token embeddings + positional encoding.

    Args:
        token_ids: (B, S) integer array of token indices
        token_embed_matrix: (vocab_size, d_model) token embedding matrix
        pos_encoding: (max_seq_len, d_model) positional encoding/embedding matrix
        use_learned_pos: If True, pos_encoding contains learned embeddings
        positions: Optional (B, S) position indices

    Returns:
        embeddings: (B, S, d_model) combined embeddings
    """
    # Get token embeddings
    token_embeds = token_embedding_forward(token_ids, token_embed_matrix)

    # Add positional encoding
    return add_positional_encoding(token_embeds, pos_encoding, positions)


def embedding_layer_backward(token_ids: np.ndarray,
                             token_embed_matrix: np.ndarray,
                             pos_encoding: np.ndarray,
                             dOut: np.ndarray,
                             use_learned_pos: bool = False,
                             positions: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward pass for complete embedding layer.

    Args:
        token_ids: (B, S) integer array of token indices
        token_embed_matrix: (vocab_size, d_model) token embedding matrix
        pos_encoding: (max_seq_len, d_model) positional encoding/embedding matrix
        dOut: (B, S, d_model) gradient from upstream
        use_learned_pos: If True, pos_encoding contains learned embeddings
        positions: Optional (B, S) position indices

    Returns:
        dTokenEmbed: (vocab_size, d_model) gradient for token embeddings
        dPosEmbed: (max_seq_len, d_model) gradient for positional embeddings
                   (None if using fixed sinusoidal encoding)
    """
    B, S = token_ids.shape

    if positions is None:
        positions = np.arange(S)[np.newaxis, :].repeat(B, axis=0)

    # Gradient flows equally to both token and positional embeddings
    dTokenEmbed = token_embedding_backward(token_ids, token_embed_matrix, dOut)

    if use_learned_pos:
        dPosEmbed = learned_positional_embedding_backward(positions, pos_encoding, dOut)
    else:
        # Fixed sinusoidal encoding - no gradient needed
        dPosEmbed = None

    return dTokenEmbed, dPosEmbed