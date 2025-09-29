"""
Download and prepare TinyStories dataset for training.
"""
import os
import urllib.request
import gzip
import json
from typing import Optional


def download_file(url: str, output_path: str):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    print(f"  -> {output_path}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if count % 100 == 0:  # Update every 100 blocks
            print(f"  Progress: {percent}%", end='\r')

    urllib.request.urlretrieve(url, output_path, progress_hook)
    print(f"\n  Done!")


def extract_gz(gz_path: str, output_path: str):
    """Extract .gz file."""
    print(f"Extracting {gz_path}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())
    print(f"  Extracted to {output_path}")


def prepare_tinystories(data_dir: str = 'data', sample_size: Optional[int] = None):
    """
    Download and prepare TinyStories dataset.

    Args:
        data_dir: Directory to store data files
        sample_size: If provided, only use first N stories (for quick testing)
    """
    os.makedirs(data_dir, exist_ok=True)

    # TinyStories dataset URLs
    # Using the validation split as it's smaller for testing
    urls = {
        'train': 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt',
        'valid': 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt'
    }

    for split, url in urls.items():
        # Download
        output_file = os.path.join(data_dir, f'tinystories_{split}.txt')

        if os.path.exists(output_file):
            print(f"{output_file} already exists, skipping download")
        else:
            try:
                download_file(url, output_file)
            except Exception as e:
                print(f"Error downloading {split}: {e}")
                print(f"Please manually download from: {url}")
                continue

        # If sample size specified, create a smaller version
        if sample_size is not None:
            sample_file = os.path.join(data_dir, f'tinystories_{split}_sample.txt')
            print(f"Creating sample with {sample_size} stories...")

            with open(output_file, 'r', encoding='utf-8') as f_in:
                with open(sample_file, 'w', encoding='utf-8') as f_out:
                    story_count = 0
                    current_story = []

                    for line in f_in:
                        line = line.strip()

                        if line.startswith('<|endoftext|>'):
                            # End of story
                            if current_story:
                                f_out.write(' '.join(current_story) + '\n')
                                story_count += 1
                                current_story = []

                                if story_count >= sample_size:
                                    break
                        elif line:
                            current_story.append(line)

            print(f"  Created sample file: {sample_file}")

    print("\nDataset preparation complete!")
    print(f"Files in {data_dir}:")
    for f in os.listdir(data_dir):
        file_path = os.path.join(data_dir, f)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {f}: {size_mb:.2f} MB")


def count_stories(file_path: str) -> int:
    """Count number of stories in a file."""
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download and prepare TinyStories dataset')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory to store data files')
    parser.add_argument('--sample_size', type=int, default=10000,
                        help='Number of stories to sample (for quick testing)')
    parser.add_argument('--full', action='store_true',
                        help='Download full dataset (no sampling)')

    args = parser.parse_args()

    sample_size = None if args.full else args.sample_size

    prepare_tinystories(args.data_dir, sample_size)

    # Count stories if sample file exists
    if sample_size is not None:
        for split in ['train', 'valid']:
            sample_file = os.path.join(args.data_dir, f'tinystories_{split}_sample.txt')
            if os.path.exists(sample_file):
                n_stories = count_stories(sample_file)
                print(f"\n{split} sample: {n_stories} stories")