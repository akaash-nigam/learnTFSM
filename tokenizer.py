"""
Tokenizer integration for TinyStories.
Supports both tiktoken and custom BPE tokenizers.
"""
import json
from typing import List, Optional
import os
import sys

# Add lecture-two to path for importing BPE tokenizer
lecture_two_path = os.path.join(os.path.dirname(__file__), '..', 'lecture-two')
sys.path.insert(0, lecture_two_path)


class TiktokenWrapper:
    """Wrapper for tiktoken (OpenAI's BPE tokenizer)."""

    def __init__(self, encoding_name: str = "gpt2"):
        """
        Initialize tiktoken tokenizer.

        Args:
            encoding_name: Name of tiktoken encoding (gpt2, r50k_base, etc.)
        """
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding(encoding_name)
            self.vocab_size = self.tokenizer.n_vocab
            print(f"Loaded tiktoken with encoding '{encoding_name}'")
            print(f"  Vocab size: {self.vocab_size}")
        except ImportError:
            raise ImportError(
                "tiktoken not installed. Install with: pip install tiktoken"
            )

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size


class Lecture2BPETokenizer:
    """
    Wrapper for the BPE tokenizer from lecture-two.
    """

    def __init__(self, model_file: str):
        """
        Initialize from model file.

        Args:
            model_file: Path to BPE model JSON file
        """
        try:
            from tokenization.EncoderDecoder import BPETokenizer
        except ImportError:
            raise ImportError(
                "Cannot import BPETokenizer from lecture-two. "
                "Make sure lecture-two/tokenization is available."
            )

        self.tokenizer = BPETokenizer()
        self.tokenizer.load_model_from_file(model_file)
        self.vocab_size = self.tokenizer.get_vocab_size()

        print(f"Loaded Lecture-2 BPE tokenizer from {model_file}")
        print(f"  Vocab size: {self.vocab_size}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size


class CharacterTokenizer:
    """
    Simple character-level tokenizer for testing.
    """

    def __init__(self, text_sample: Optional[str] = None):
        """
        Initialize character tokenizer.

        Args:
            text_sample: Optional text sample to build vocabulary from
        """
        # Basic ASCII + special tokens
        chars = set(chr(i) for i in range(32, 127))  # Printable ASCII

        if text_sample:
            chars.update(set(text_sample))

        # Add special tokens
        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']

        # Create vocab
        self.vocab = {token: i for i, token in enumerate(special_tokens)}
        for i, char in enumerate(sorted(chars)):
            self.vocab[char] = len(self.vocab)

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        print(f"Created character tokenizer")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Characters: {len(self.vocab) - len(special_tokens)}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.vocab.get(char, self.vocab['<unk>']) for char in text]

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join(self.inv_vocab.get(tid, '<unk>') for tid in token_ids)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size


def get_tokenizer(tokenizer_type: str = "tiktoken",
                  model_file: Optional[str] = None,
                  encoding_name: str = "gpt2") -> object:
    """
    Get a tokenizer instance.

    Args:
        tokenizer_type: Type of tokenizer ('tiktoken', 'lecture2_bpe', 'char')
        model_file: Path to model file (for 'lecture2_bpe' type)
        encoding_name: Encoding name for tiktoken

    Returns:
        Tokenizer instance with encode/decode/get_vocab_size methods
    """
    if tokenizer_type == "tiktoken":
        return TiktokenWrapper(encoding_name)
    elif tokenizer_type == "lecture2_bpe":
        if model_file is None:
            raise ValueError("model_file required for lecture2_bpe tokenizer")
        return Lecture2BPETokenizer(model_file)
    elif tokenizer_type == "char":
        return CharacterTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def tokenize_file(input_file: str, output_file: str, tokenizer,
                  max_length: Optional[int] = None):
    """
    Tokenize a text file and save as jsonl.

    Args:
        input_file: Input text file (one story per line)
        output_file: Output jsonl file
        tokenizer: Tokenizer instance
        max_length: Optional maximum sequence length (truncate if longer)
    """
    print(f"Tokenizing {input_file}...")

    n_lines = 0
    n_tokens = 0

    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w') as f_out:
            for i, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    continue

                # Tokenize
                token_ids = tokenizer.encode(line)

                # Optionally truncate
                if max_length is not None and len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]

                # Skip very short sequences
                if len(token_ids) < 2:
                    continue

                # Write as JSON
                f_out.write(json.dumps(token_ids) + '\n')

                n_lines += 1
                n_tokens += len(token_ids)

                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1} lines...")

    print(f"Done! Tokenized {n_lines} sequences")
    print(f"  Total tokens: {n_tokens:,}")
    print(f"  Average length: {n_tokens / max(n_lines, 1):.1f}")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Tokenize TinyStories data')
    parser.add_argument('input_file', type=str, help='Input text file')
    parser.add_argument('output_file', type=str, help='Output jsonl file')
    parser.add_argument('--tokenizer', type=str, default='tiktoken',
                        choices=['tiktoken', 'lecture2_bpe', 'char'],
                        help='Tokenizer type')
    parser.add_argument('--model_file', type=str, default=None,
                        help='Model file for lecture2_bpe tokenizer')
    parser.add_argument('--encoding', type=str, default='gpt2',
                        help='Encoding name for tiktoken')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Maximum sequence length (truncate if longer)')

    args = parser.parse_args()

    # Get tokenizer
    tokenizer = get_tokenizer(
        tokenizer_type=args.tokenizer,
        model_file=args.model_file,
        encoding_name=args.encoding
    )

    # Tokenize file
    tokenize_file(args.input_file, args.output_file, tokenizer, args.max_length)