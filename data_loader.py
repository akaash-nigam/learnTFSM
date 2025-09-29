"""
Data loader for TinyStories dataset.
"""
import numpy as np
from typing import List, Tuple, Optional, Iterator
import json
import os


class TinyStoriesDataset:
    """
    Dataset class for TinyStories.

    Expects tokenized data files with one story per line, where each line is
    a JSON array of token IDs.
    """

    def __init__(self, file_path: str, max_seq_len: int = 256):
        """
        Initialize dataset.

        Args:
            file_path: Path to tokenized data file (jsonl format)
            max_seq_len: Maximum sequence length for training
        """
        self.file_path = file_path
        self.max_seq_len = max_seq_len
        self.data = []

        self._load_data()

    def _load_data(self):
        """Load tokenized data from file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        print(f"Loading data from {self.file_path}...")

        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse JSON array of token IDs
                try:
                    token_ids = json.loads(line)
                    if isinstance(token_ids, list) and len(token_ids) > 1:
                        self.data.append(token_ids)
                except json.JSONDecodeError:
                    continue

        print(f"Loaded {len(self.data)} sequences")

    def __len__(self) -> int:
        """Return number of sequences in dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> List[int]:
        """Get a single sequence by index."""
        return self.data[idx]


class DataLoader:
    """
    Data loader that creates batches for language model training.

    Creates input-target pairs where target is shifted by 1 position
    for next-token prediction.
    """

    def __init__(self, dataset: TinyStoriesDataset, batch_size: int,
                 seq_len: int, shuffle: bool = True, drop_last: bool = True):
        """
        Initialize data loader.

        Args:
            dataset: TinyStoriesDataset instance
            batch_size: Number of sequences per batch
            seq_len: Length of each sequence (will truncate/pad)
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(self.indices)

        self.current_idx = 0

    def __len__(self) -> int:
        """Return number of batches."""
        n_samples = len(self.dataset)
        if self.drop_last:
            return n_samples // self.batch_size
        else:
            return (n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over batches."""
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get next batch."""
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        # Get batch indices
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]

        if len(batch_indices) < self.batch_size and self.drop_last:
            raise StopIteration

        self.current_idx += self.batch_size

        # Create batch
        return self._create_batch(batch_indices)

    def _create_batch(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a batch of input-target pairs.

        Args:
            indices: Indices of sequences to include in batch

        Returns:
            inputs: (B, seq_len) array of input token IDs
            targets: (B, seq_len) array of target token IDs (shifted by 1)
        """
        batch_size = len(indices)
        inputs = np.zeros((batch_size, self.seq_len), dtype=np.int32)
        targets = np.zeros((batch_size, self.seq_len), dtype=np.int32)

        for i, idx in enumerate(indices):
            tokens = self.dataset[idx]

            # We need seq_len + 1 tokens (input + target)
            if len(tokens) >= self.seq_len + 1:
                # Truncate
                inputs[i] = tokens[:self.seq_len]
                targets[i] = tokens[1:self.seq_len + 1]
            else:
                # Pad with zeros (we'll use ignore_index in loss)
                # For now, just skip sequences that are too short
                # In production, you'd handle padding properly
                available_len = min(len(tokens) - 1, self.seq_len)
                if available_len > 0:
                    inputs[i, :available_len] = tokens[:available_len]
                    targets[i, :available_len] = tokens[1:available_len + 1]
                    # Mark rest as padding
                    targets[i, available_len:] = -1  # ignore_index

        return inputs, targets


class InfiniteDataLoader:
    """
    Data loader that loops infinitely over the dataset.
    Useful for training with a fixed number of steps.
    """

    def __init__(self, dataset: TinyStoriesDataset, batch_size: int,
                 seq_len: int, shuffle: bool = True):
        """
        Initialize infinite data loader.

        Args:
            dataset: TinyStoriesDataset instance
            batch_size: Number of sequences per batch
            seq_len: Length of each sequence
            shuffle: Whether to shuffle after each epoch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.shuffle = shuffle

        self.indices = np.arange(len(dataset))
        if shuffle:
            np.random.shuffle(self.indices)

        self.current_idx = 0

    def __iter__(self):
        """Return self as iterator."""
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get next batch, looping infinitely."""
        # Check if we need to reset
        if self.current_idx + self.batch_size > len(self.dataset):
            self.current_idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)

        # Get batch indices
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size

        # Create batch
        batch_size = len(batch_indices)
        inputs = np.zeros((batch_size, self.seq_len), dtype=np.int32)
        targets = np.zeros((batch_size, self.seq_len), dtype=np.int32)

        for i, idx in enumerate(batch_indices):
            tokens = self.dataset[idx]

            # We need seq_len + 1 tokens
            if len(tokens) >= self.seq_len + 1:
                inputs[i] = tokens[:self.seq_len]
                targets[i] = tokens[1:self.seq_len + 1]
            else:
                available_len = min(len(tokens) - 1, self.seq_len)
                if available_len > 0:
                    inputs[i, :available_len] = tokens[:available_len]
                    targets[i, :available_len] = tokens[1:available_len + 1]
                    targets[i, available_len:] = -1

        return inputs, targets


def prepare_data_file(input_file: str, output_file: str, tokenize_fn):
    """
    Prepare tokenized data file from raw text.

    Args:
        input_file: Path to input text file (one story per line)
        output_file: Path to output jsonl file
        tokenize_fn: Function that takes text and returns list of token IDs
    """
    print(f"Preparing data from {input_file} -> {output_file}")

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for i, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue

            # Tokenize
            token_ids = tokenize_fn(line)

            # Write as JSON array
            f_out.write(json.dumps(token_ids) + '\n')

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1} stories...")

    print(f"Done! Wrote to {output_file}")


# Simple test
if __name__ == "__main__":
    print("Data loader module - use this to load TinyStories data")
    print()
    print("Example usage:")
    print("""
    # Load dataset
    dataset = TinyStoriesDataset('train_tokenized.jsonl', max_seq_len=256)

    # Create data loader
    loader = DataLoader(dataset, batch_size=32, seq_len=128, shuffle=True)

    # Iterate over batches
    for inputs, targets in loader:
        # inputs: (32, 128) - input tokens
        # targets: (32, 128) - target tokens (shifted by 1)
        print(inputs.shape, targets.shape)
        break
    """)