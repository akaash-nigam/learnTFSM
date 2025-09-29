"""
Complete NumPy language model implementation.
Assembles embeddings, transformer blocks, and language model head.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from embeddings import (
    embedding_layer_forward,
    embedding_layer_backward,
    sinusoidal_positional_encoding
)
from transformer import (
    multi_head_attention,
    multi_head_attention_backward,
    layer_norm,
    layer_norm_backward,
    feed_forward_network,
    feed_forward_network_backward,
    qkv_projection,
    matmul_backward
)
from lm_head import lm_head_forward, lm_head_backward, tied_lm_head_forward, tied_lm_head_backward
from loss import cross_entropy_loss_with_logits


class TransformerLM:
    """
    A complete Transformer language model implemented in NumPy.

    Architecture:
    - Token embeddings + positional encoding
    - N transformer blocks (attention + FFN with residual connections)
    - Language model head (projects to vocabulary)
    - Cross-entropy loss for next-token prediction
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 n_layers: int, max_seq_len: int = 512,
                 use_learned_pos: bool = False, tie_embeddings: bool = True,
                 dropout: float = 0.0):
        """
        Initialize the language model.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            d_ff: Feedforward hidden dimension
            n_layers: Number of transformer blocks
            max_seq_len: Maximum sequence length
            use_learned_pos: Use learned positional embeddings (vs sinusoidal)
            tie_embeddings: Tie input and output embeddings
            dropout: Dropout rate (not implemented yet)
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_learned_pos = use_learned_pos
        self.tie_embeddings = tie_embeddings

        # Dimension per head
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads

        # Initialize parameters
        self.params = self._init_parameters()

    def _init_parameters(self) -> Dict[str, np.ndarray]:
        """Initialize all model parameters."""
        params = {}

        # Token embeddings
        params['token_embed'] = np.random.randn(self.vocab_size, self.d_model) * 0.02

        # Positional encoding/embeddings
        if self.use_learned_pos:
            params['pos_embed'] = np.random.randn(self.max_seq_len, self.d_model) * 0.02
        else:
            # Fixed sinusoidal - not a parameter
            params['pos_encoding'] = sinusoidal_positional_encoding(self.max_seq_len, self.d_model)

        # Transformer blocks
        for layer in range(self.n_layers):
            prefix = f'layer_{layer}'

            # Attention parameters (for all heads combined)
            d_qkv = self.n_heads * self.d_head
            params[f'{prefix}_W_Q'] = np.random.randn(self.d_model, d_qkv) * np.sqrt(2.0 / self.d_model)
            params[f'{prefix}_W_K'] = np.random.randn(self.d_model, d_qkv) * np.sqrt(2.0 / self.d_model)
            params[f'{prefix}_W_V'] = np.random.randn(self.d_model, d_qkv) * np.sqrt(2.0 / self.d_model)
            params[f'{prefix}_W_O'] = np.random.randn(d_qkv, self.d_model) * np.sqrt(2.0 / d_qkv)

            # Feedforward parameters
            params[f'{prefix}_W_FF1'] = np.random.randn(self.d_model, self.d_ff) * np.sqrt(2.0 / self.d_model)
            params[f'{prefix}_W_FF2'] = np.random.randn(self.d_ff, self.d_model) * np.sqrt(2.0 / self.d_ff)

            # Layer norm parameters (2 per block: after attention, after FFN)
            params[f'{prefix}_ln1_gamma'] = np.ones(self.d_model)
            params[f'{prefix}_ln1_beta'] = np.zeros(self.d_model)
            params[f'{prefix}_ln2_gamma'] = np.ones(self.d_model)
            params[f'{prefix}_ln2_beta'] = np.zeros(self.d_model)

        # Language model head
        if not self.tie_embeddings:
            params['lm_head_W'] = np.random.randn(self.d_model, self.vocab_size) * 0.02

        return params

    def forward(self, token_ids: np.ndarray, cache: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass through the model.

        Args:
            token_ids: (B, S) integer array of token indices
            cache: Optional cache dict for storing intermediate activations

        Returns:
            logits: (B, S, vocab_size) next-token prediction logits
            cache: Dictionary of cached activations for backward pass
        """
        if cache is None:
            cache = {}

        B, S = token_ids.shape
        cache['token_ids'] = token_ids

        # Embeddings
        if self.use_learned_pos:
            pos_enc_matrix = self.params['pos_embed']
        else:
            pos_enc_matrix = self.params['pos_encoding']
        x = embedding_layer_forward(token_ids, self.params['token_embed'],
                                    pos_enc_matrix, self.use_learned_pos)
        cache['embedded'] = x

        # Transformer blocks
        for layer in range(self.n_layers):
            x = self._forward_block(x, layer, cache)

        cache['final_hidden'] = x

        # Language model head
        if self.tie_embeddings:
            logits = tied_lm_head_forward(x, self.params['token_embed'])
        else:
            logits = lm_head_forward(x, self.params['lm_head_W'], b_lm=None)

        cache['logits'] = logits

        return logits, cache

    def _forward_block(self, x: np.ndarray, layer: int, cache: Dict) -> np.ndarray:
        """Forward pass through a single transformer block."""
        prefix = f'layer_{layer}'
        cache[f'{prefix}_input'] = x

        # Get QKV projections
        q, k, v = qkv_projection(x,
                                  self.params[f'{prefix}_W_Q'],
                                  self.params[f'{prefix}_W_K'],
                                  self.params[f'{prefix}_W_V'])

        # Reshape for multi-head attention: (B, S, d_qkv) -> (B, H, S, d_head)
        B, S, d_qkv = q.shape
        q = q.reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        cache[f'{prefix}_q'] = q
        cache[f'{prefix}_k'] = k
        cache[f'{prefix}_v'] = v

        # Multi-head attention with causal masking
        attn_out = multi_head_attention(q, k, v, causal=True)  # (B, H, S, d_head)

        # Reshape back: (B, H, S, d_head) -> (B, S, d_qkv)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_qkv)
        cache[f'{prefix}_attn_out'] = attn_out

        # Output projection
        attn_proj = attn_out @ self.params[f'{prefix}_W_O']
        cache[f'{prefix}_attn_proj'] = attn_proj

        # First residual connection + layer norm
        x = x + attn_proj
        cache[f'{prefix}_after_attn_residual'] = x
        x = layer_norm(x, self.params[f'{prefix}_ln1_gamma'], self.params[f'{prefix}_ln1_beta'])
        cache[f'{prefix}_after_ln1'] = x

        # Feedforward
        ff_out = feed_forward_network(x, self.params[f'{prefix}_W_FF1'],
                                      self.params[f'{prefix}_W_FF2'])
        cache[f'{prefix}_ff_out'] = ff_out

        # Second residual connection + layer norm
        x = x + ff_out
        cache[f'{prefix}_after_ff_residual'] = x
        x = layer_norm(x, self.params[f'{prefix}_ln2_gamma'], self.params[f'{prefix}_ln2_beta'])
        cache[f'{prefix}_output'] = x

        return x

    def backward(self, targets: np.ndarray, cache: Dict,
                 ignore_index: int = -1) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Backward pass through the model.

        Args:
            targets: (B, S) integer array of target token indices
            cache: Dictionary of cached activations from forward pass
            ignore_index: Target value to ignore (padding tokens)

        Returns:
            loss: Scalar loss value
            grads: Dictionary of gradients for all parameters
        """
        grads = {}

        # Compute loss and gradient w.r.t. logits
        logits = cache['logits']
        loss, dLogits = cross_entropy_loss_with_logits(logits, targets, ignore_index)

        # Backprop through language model head
        final_hidden = cache['final_hidden']
        if self.tie_embeddings:
            dHidden, dEmbed_from_lm = tied_lm_head_backward(final_hidden, self.params['token_embed'], dLogits)
        else:
            dHidden, grads['lm_head_W'], _ = lm_head_backward(final_hidden, self.params['lm_head_W'], dLogits)
            dEmbed_from_lm = None

        # Backprop through transformer blocks (reverse order)
        for layer in reversed(range(self.n_layers)):
            dHidden = self._backward_block(dHidden, layer, cache, grads)

        # Backprop through embeddings
        token_ids = cache['token_ids']
        if self.use_learned_pos:
            pos_enc_matrix = self.params['pos_embed']
        else:
            pos_enc_matrix = self.params['pos_encoding']
        dTokenEmbed, dPosEmbed = embedding_layer_backward(
            token_ids, self.params['token_embed'], pos_enc_matrix,
            dHidden, self.use_learned_pos
        )

        # Accumulate embedding gradients
        if self.tie_embeddings and dEmbed_from_lm is not None:
            grads['token_embed'] = dTokenEmbed + dEmbed_from_lm
        else:
            grads['token_embed'] = dTokenEmbed

        if self.use_learned_pos and dPosEmbed is not None:
            grads['pos_embed'] = dPosEmbed

        return loss, grads

    def _backward_block(self, dOut: np.ndarray, layer: int, cache: Dict, grads: Dict) -> np.ndarray:
        """Backward pass through a single transformer block."""
        prefix = f'layer_{layer}'

        # Backprop through second layer norm
        after_ff_residual = cache[f'{prefix}_after_ff_residual']
        dResidual2, dGamma2, dBeta2 = layer_norm_backward(
            after_ff_residual,
            self.params[f'{prefix}_ln2_gamma'],
            self.params[f'{prefix}_ln2_beta'],
            dOut
        )
        grads[f'{prefix}_ln2_gamma'] = dGamma2
        grads[f'{prefix}_ln2_beta'] = dBeta2

        # Backprop through second residual connection
        dFF = dResidual2
        dAfterLN1 = dResidual2

        # Backprop through feedforward
        after_ln1 = cache[f'{prefix}_after_ln1']
        dFFInput, dW_FF1, dW_FF2 = feed_forward_network_backward(
            after_ln1,
            self.params[f'{prefix}_W_FF1'],
            self.params[f'{prefix}_W_FF2'],
            dFF
        )
        grads[f'{prefix}_W_FF1'] = dW_FF1
        grads[f'{prefix}_W_FF2'] = dW_FF2

        dAfterLN1 += dFFInput

        # Backprop through first layer norm
        after_attn_residual = cache[f'{prefix}_after_attn_residual']
        dResidual1, dGamma1, dBeta1 = layer_norm_backward(
            after_attn_residual,
            self.params[f'{prefix}_ln1_gamma'],
            self.params[f'{prefix}_ln1_beta'],
            dAfterLN1
        )
        grads[f'{prefix}_ln1_gamma'] = dGamma1
        grads[f'{prefix}_ln1_beta'] = dBeta1

        # Backprop through first residual connection
        dAttnProj = dResidual1
        dInput = dResidual1

        # Backprop through output projection
        attn_out = cache[f'{prefix}_attn_out']
        B, S, d_qkv = attn_out.shape
        dAttnOut_2d, dW_O = matmul_backward(
            attn_out.reshape(-1, d_qkv),
            self.params[f'{prefix}_W_O'],
            dAttnProj.reshape(-1, self.d_model)
        )
        grads[f'{prefix}_W_O'] = dW_O
        dAttnOut = dAttnOut_2d.reshape(B, S, d_qkv)

        # Reshape to multi-head format
        dAttnOut = dAttnOut.reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        # Backprop through attention
        q = cache[f'{prefix}_q']
        k = cache[f'{prefix}_k']
        v = cache[f'{prefix}_v']
        dQ, dK, dV = multi_head_attention_backward(q, k, v, dAttnOut, causal=True)

        # Reshape back to (B, S, d_qkv)
        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, S, d_qkv)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, S, d_qkv)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, S, d_qkv)

        # Backprop through QKV projections
        x_input = cache[f'{prefix}_input']
        x_2d = x_input.reshape(-1, self.d_model)

        dXq_2d, dW_Q = matmul_backward(x_2d, self.params[f'{prefix}_W_Q'], dQ.reshape(-1, d_qkv))
        dXk_2d, dW_K = matmul_backward(x_2d, self.params[f'{prefix}_W_K'], dK.reshape(-1, d_qkv))
        dXv_2d, dW_V = matmul_backward(x_2d, self.params[f'{prefix}_W_V'], dV.reshape(-1, d_qkv))

        grads[f'{prefix}_W_Q'] = dW_Q
        grads[f'{prefix}_W_K'] = dW_K
        grads[f'{prefix}_W_V'] = dW_V

        dX_from_qkv = (dXq_2d + dXk_2d + dXv_2d).reshape(B, S, self.d_model)
        dInput += dX_from_qkv

        return dInput

    def get_num_params(self) -> int:
        """Count total number of parameters."""
        total = 0
        for name, param in self.params.items():
            if name != 'pos_encoding':  # Don't count fixed positional encoding
                total += param.size
        return total