"""
Plot training history and metrics.
"""
import json
import numpy as np
import os


def plot_training_history(history_file: str, output_dir: Optional[str] = None):
    """
    Plot training history from JSON file.

    Args:
        history_file: Path to training_history.json
        output_dir: Optional directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)

    train_losses = history['train_losses']
    val_losses = history['val_losses']
    val_perplexities = history['val_perplexities']

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Training loss
    axes[0].plot(train_losses, alpha=0.5, linewidth=0.5)
    # Add smoothed version
    if len(train_losses) > 100:
        window = min(100, len(train_losses) // 10)
        smoothed = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window//2, window//2 + len(smoothed)), smoothed, linewidth=2)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Validation loss
    if val_losses:
        val_steps = [x[0] for x in val_losses]
        val_loss_values = [x[1] for x in val_losses]
        axes[1].plot(val_steps, val_loss_values, marker='o', linewidth=2)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation Loss')
        axes[1].grid(True, alpha=0.3)

    # Plot 3: Validation perplexity
    if val_perplexities:
        val_steps = [x[0] for x in val_perplexities]
        val_ppl_values = [x[1] for x in val_perplexities]
        axes[2].plot(val_steps, val_ppl_values, marker='o', linewidth=2, color='green')
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Perplexity')
        axes[2].set_title('Validation Perplexity')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def print_training_summary(history_file: str):
    """
    Print summary statistics from training.

    Args:
        history_file: Path to training_history.json
    """
    with open(history_file, 'r') as f:
        history = json.load(f)

    train_losses = history['train_losses']
    val_losses = history.get('val_losses', [])
    val_perplexities = history.get('val_perplexities', [])
    total_steps = history.get('total_steps', len(train_losses))

    print("="*60)
    print("Training Summary")
    print("="*60)
    print(f"Total steps: {total_steps:,}")
    print()

    print("Training Loss:")
    print(f"  Initial: {train_losses[0]:.4f}")
    print(f"  Final: {train_losses[-1]:.4f}")
    print(f"  Final (avg last 100): {np.mean(train_losses[-100:]):.4f}")
    print()

    if val_losses:
        val_loss_values = [x[1] for x in val_losses]
        print("Validation Loss:")
        print(f"  Initial: {val_loss_values[0]:.4f}")
        print(f"  Final: {val_loss_values[-1]:.4f}")
        print(f"  Best: {min(val_loss_values):.4f}")
        print()

    if val_perplexities:
        val_ppl_values = [x[1] for x in val_perplexities]
        print("Validation Perplexity:")
        print(f"  Initial: {val_ppl_values[0]:.2f}")
        print(f"  Final: {val_ppl_values[-1]:.2f}")
        print(f"  Best: {min(val_ppl_values):.2f}")
        print()

    print("="*60)


def main():
    """Main function."""
    import argparse
    from typing import Optional

    parser = argparse.ArgumentParser(description='Plot training history')
    parser.add_argument('history_file', type=str,
                        help='Path to training_history.json')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots (if not specified, displays plot)')
    parser.add_argument('--summary_only', action='store_true',
                        help='Only print summary statistics')

    args = parser.parse_args()

    # Print summary
    print_training_summary(args.history_file)
    print()

    # Plot (if not summary only)
    if not args.summary_only:
        plot_training_history(args.history_file, args.output_dir)


if __name__ == "__main__":
    main()