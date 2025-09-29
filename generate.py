"""
Text generation with trained NumPy Transformer.
"""
import numpy as np
from typing import List, Optional
from language_model import TransformerLM


def sample_token(logits: np.ndarray, temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> int:
    """
    Sample next token from logits.

    Args:
        logits: (vocab_size,) logits for next token
        temperature: Sampling temperature (higher = more random)
        top_k: If provided, only sample from top k tokens
        top_p: If provided, nucleus sampling (sample from top p probability mass)

    Returns:
        token_id: Sampled token ID
    """
    # Apply temperature
    logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None:
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = -np.inf

    # Softmax to get probabilities
    logits_max = logits.max()
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum()

    # Apply top-p (nucleus) filtering
    if top_p is not None:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        if sorted_indices_to_remove.sum() == len(sorted_indices_to_remove):
            sorted_indices_to_remove[-1] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0
        probs = probs / probs.sum()

    # Sample
    token_id = np.random.choice(len(probs), p=probs)
    return token_id


def generate(model: TransformerLM, prompt_ids: List[int], max_length: int = 100,
             temperature: float = 1.0, top_k: Optional[int] = None,
             top_p: Optional[float] = None, eos_token_id: Optional[int] = None) -> List[int]:
    """
    Generate text continuation from prompt.

    Args:
        model: Trained TransformerLM
        prompt_ids: List of token IDs for prompt
        max_length: Maximum length to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        eos_token_id: End-of-sequence token ID (stop generation if sampled)

    Returns:
        generated_ids: List of generated token IDs (includes prompt)
    """
    generated = list(prompt_ids)

    for _ in range(max_length):
        # Prepare input (use last max_seq_len tokens if longer)
        if len(generated) > model.max_seq_len:
            input_ids = np.array([generated[-model.max_seq_len:]], dtype=np.int32)
        else:
            input_ids = np.array([generated], dtype=np.int32)

        # Forward pass
        logits, _ = model.forward(input_ids)

        # Get logits for next token (last position)
        next_token_logits = logits[0, -1, :]

        # Sample
        next_token = sample_token(next_token_logits, temperature, top_k, top_p)
        generated.append(int(next_token))

        # Check for end-of-sequence
        if eos_token_id is not None and next_token == eos_token_id:
            break

    return generated


def generate_batch(model: TransformerLM, prompt_ids_batch: List[List[int]],
                   max_length: int = 100, temperature: float = 1.0,
                   top_k: Optional[int] = None, top_p: Optional[float] = None,
                   eos_token_id: Optional[int] = None) -> List[List[int]]:
    """
    Generate text for a batch of prompts.

    Args:
        model: Trained TransformerLM
        prompt_ids_batch: List of prompt token ID lists
        max_length: Maximum length to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        eos_token_id: End-of-sequence token ID

    Returns:
        generated_batch: List of generated token ID lists
    """
    return [
        generate(model, prompt_ids, max_length, temperature, top_k, top_p, eos_token_id)
        for prompt_ids in prompt_ids_batch
    ]


def load_model(checkpoint_path: str, vocab_size: int, d_model: int, n_heads: int,
               d_ff: int, n_layers: int, max_seq_len: int = 512,
               use_learned_pos: bool = False, tie_embeddings: bool = True) -> TransformerLM:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to .npz checkpoint file
        vocab_size: Model vocabulary size
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feedforward dimension
        n_layers: Number of layers
        max_seq_len: Maximum sequence length
        use_learned_pos: Whether model uses learned positional embeddings
        tie_embeddings: Whether model ties embeddings

    Returns:
        model: Loaded TransformerLM
    """
    # Create model
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        use_learned_pos=use_learned_pos,
        tie_embeddings=tie_embeddings
    )

    # Load parameters
    checkpoint = np.load(checkpoint_path)
    for key in checkpoint.files:
        if key in model.params:
            model.params[key] = checkpoint[key]

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Parameters: {model.get_num_params():,}")

    return model


def main():
    """Interactive text generation."""
    import argparse
    from tokenizer import get_tokenizer

    parser = argparse.ArgumentParser(description='Generate text with trained model')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vocab_size', type=int, default=50257,
                        help='Vocabulary size')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=512,
                        help='Feedforward dimension')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--max_seq_len', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--use_learned_pos', action='store_true',
                        help='Use learned positional embeddings')
    parser.add_argument('--no_tie_embeddings', action='store_true',
                        help='Do not tie embeddings')

    # Tokenizer arguments
    parser.add_argument('--tokenizer_type', type=str, default='tiktoken',
                        choices=['tiktoken', 'lecture2_bpe', 'char'],
                        help='Tokenizer type')
    parser.add_argument('--tokenizer_model', type=str, default=None,
                        help='Path to tokenizer model file')
    parser.add_argument('--encoding', type=str, default='gpt2',
                        help='Tiktoken encoding name')

    # Generation arguments
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                        help='Prompt text')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-k filtering')
    parser.add_argument('--top_p', type=float, default=None,
                        help='Nucleus sampling threshold')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)

    print("="*60)
    print("Text Generation with NumPy Transformer")
    print("="*60)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(
        tokenizer_type=args.tokenizer_type,
        model_file=args.tokenizer_model,
        encoding_name=args.encoding
    )
    print()

    # Load model
    print("Loading model...")
    model = load_model(
        checkpoint_path=args.checkpoint,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        use_learned_pos=args.use_learned_pos,
        tie_embeddings=not args.no_tie_embeddings
    )
    print()

    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"Prompt: {args.prompt}")
    print(f"Prompt tokens: {len(prompt_ids)}")
    print()

    # Generate
    print("Generating...")
    print("="*60)
    print()

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"Sample {i+1}:")

        generated_ids = generate(
            model=model,
            prompt_ids=prompt_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )

        generated_text = tokenizer.decode(generated_ids)
        print(generated_text)
        print()

    print("="*60)


if __name__ == "__main__":
    main()