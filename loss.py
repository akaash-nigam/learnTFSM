"""
Loss functions for language modeling.
"""
import numpy as np
from typing import Tuple


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray,
                       ignore_index: int = -1) -> float:
    """
    Compute cross-entropy loss for language modeling.

    Args:
        logits: (B, S, vocab_size) unnormalized logits
        targets: (B, S) integer array of target token indices
        ignore_index: Target value to ignore (e.g., padding tokens)

    Returns:
        loss: scalar loss value
    """
    B, S, vocab_size = logits.shape

    # Flatten to (B*S, vocab_size) and (B*S,)
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Create mask for valid (non-ignored) positions
    mask = (targets_flat != ignore_index)
    num_valid = mask.sum()

    if num_valid == 0:
        return 0.0

    # Compute log softmax for numerical stability
    # log_softmax(x) = x - log(sum(exp(x)))
    logits_max = logits_flat.max(axis=-1, keepdims=True)
    logits_shifted = logits_flat - logits_max
    exp_logits = np.exp(logits_shifted)
    log_sum_exp = np.log(exp_logits.sum(axis=-1, keepdims=True))
    log_probs = logits_shifted - log_sum_exp

    # Gather log probabilities for target tokens
    # log_probs[i, targets_flat[i]] for each i
    target_log_probs = log_probs[np.arange(len(targets_flat)), targets_flat]

    # Apply mask and compute mean loss
    masked_log_probs = target_log_probs * mask
    loss = -masked_log_probs.sum() / num_valid

    return loss


def cross_entropy_loss_backward(logits: np.ndarray, targets: np.ndarray,
                                ignore_index: int = -1) -> np.ndarray:
    """
    Backward pass for cross-entropy loss.

    Args:
        logits: (B, S, vocab_size) unnormalized logits
        targets: (B, S) integer array of target token indices
        ignore_index: Target value to ignore (e.g., padding tokens)

    Returns:
        dLogits: (B, S, vocab_size) gradient w.r.t. logits
    """
    B, S, vocab_size = logits.shape

    # Flatten
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Create mask for valid positions
    mask = (targets_flat != ignore_index)
    num_valid = mask.sum()

    if num_valid == 0:
        return np.zeros_like(logits)

    # Compute softmax probabilities
    logits_max = logits_flat.max(axis=-1, keepdims=True)
    logits_shifted = logits_flat - logits_max
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # Gradient of cross-entropy w.r.t. logits: probs - one_hot(targets)
    dLogits_flat = probs.copy()
    dLogits_flat[np.arange(len(targets_flat)), targets_flat] -= 1.0

    # Apply mask and normalize by number of valid tokens
    dLogits_flat *= mask[:, np.newaxis]
    dLogits_flat /= num_valid

    # Reshape back to original shape
    dLogits = dLogits_flat.reshape(B, S, vocab_size)

    return dLogits


def cross_entropy_loss_with_logits(logits: np.ndarray, targets: np.ndarray,
                                   ignore_index: int = -1) -> Tuple[float, np.ndarray]:
    """
    Compute cross-entropy loss and gradient in one pass (more efficient).

    Args:
        logits: (B, S, vocab_size) unnormalized logits
        targets: (B, S) integer array of target token indices
        ignore_index: Target value to ignore (e.g., padding tokens)

    Returns:
        loss: scalar loss value
        dLogits: (B, S, vocab_size) gradient w.r.t. logits
    """
    B, S, vocab_size = logits.shape

    # Flatten
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Create mask for valid positions
    mask = (targets_flat != ignore_index)
    num_valid = mask.sum()

    if num_valid == 0:
        return 0.0, np.zeros_like(logits)

    # Compute log softmax for loss
    logits_max = logits_flat.max(axis=-1, keepdims=True)
    logits_shifted = logits_flat - logits_max
    exp_logits = np.exp(logits_shifted)
    sum_exp = exp_logits.sum(axis=-1, keepdims=True)
    log_sum_exp = np.log(sum_exp)
    log_probs = logits_shifted - log_sum_exp

    # Compute softmax for gradient
    probs = exp_logits / sum_exp

    # Loss: -mean(log_probs[targets])
    target_log_probs = log_probs[np.arange(len(targets_flat)), targets_flat]
    masked_log_probs = target_log_probs * mask
    loss = -masked_log_probs.sum() / num_valid

    # Gradient: probs - one_hot(targets)
    dLogits_flat = probs.copy()
    dLogits_flat[np.arange(len(targets_flat)), targets_flat] -= 1.0
    dLogits_flat *= mask[:, np.newaxis]
    dLogits_flat /= num_valid

    dLogits = dLogits_flat.reshape(B, S, vocab_size)

    return loss, dLogits


def perplexity(logits: np.ndarray, targets: np.ndarray,
               ignore_index: int = -1) -> float:
    """
    Compute perplexity: exp(cross_entropy_loss).

    Args:
        logits: (B, S, vocab_size) unnormalized logits
        targets: (B, S) integer array of target token indices
        ignore_index: Target value to ignore

    Returns:
        ppl: perplexity value
    """
    loss = cross_entropy_loss(logits, targets, ignore_index)
    return np.exp(loss)


def accuracy(logits: np.ndarray, targets: np.ndarray,
            ignore_index: int = -1) -> float:
    """
    Compute token-level accuracy.

    Args:
        logits: (B, S, vocab_size) unnormalized logits
        targets: (B, S) integer array of target token indices
        ignore_index: Target value to ignore

    Returns:
        acc: accuracy value (0 to 1)
    """
    # Get predictions
    predictions = logits.argmax(axis=-1)  # (B, S)

    # Create mask for valid positions
    mask = (targets != ignore_index)

    if mask.sum() == 0:
        return 0.0

    # Compare predictions to targets
    correct = (predictions == targets) & mask
    accuracy = correct.sum() / mask.sum()

    return accuracy