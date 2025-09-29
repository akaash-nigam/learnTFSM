"""
Training script for NumPy Transformer language model on TinyStories.
"""
import numpy as np
import os
import json
import time
from typing import Dict, List, Optional, Tuple

from language_model import TransformerLM
from optimizer import Adam, AdamW, LRScheduler
from data_loader import TinyStoriesDataset, InfiniteDataLoader
from tokenizer import get_tokenizer
from loss import perplexity, accuracy


class Trainer:
    """Trainer for language model."""

    def __init__(self, model: TransformerLM, optimizer, lr_scheduler,
                 train_loader, val_loader,
                 save_dir: str = 'checkpoints',
                 log_interval: int = 100,
                 eval_interval: int = 1000,
                 save_interval: int = 5000):
        """
        Initialize trainer.

        Args:
            model: TransformerLM instance
            optimizer: Optimizer instance
            lr_scheduler: Learning rate scheduler
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save checkpoints
            log_interval: Steps between logging
            eval_interval: Steps between evaluation
            save_interval: Steps between checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        os.makedirs(save_dir, exist_ok=True)

        # Training state
        self.step = 0
        self.train_losses = []
        self.val_losses = []
        self.val_perplexities = []

    def train_step(self, inputs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
        """
        Perform a single training step.

        Args:
            inputs: (B, S) input token IDs
            targets: (B, S) target token IDs

        Returns:
            loss: Scalar loss value
            acc: Accuracy
        """
        # Forward pass
        logits, cache = self.model.forward(inputs)

        # Backward pass (computes loss and gradients)
        loss, grads = self.model.backward(targets, cache, ignore_index=-1)

        # Check for NaN
        if np.isnan(loss):
            print("WARNING: NaN loss detected!")
            return loss, 0.0

        # Gradient clipping
        max_grad_norm = 1.0
        total_norm = 0.0
        for grad in grads.values():
            total_norm += (grad ** 2).sum()
        total_norm = np.sqrt(total_norm)

        if total_norm > max_grad_norm:
            scale = max_grad_norm / (total_norm + 1e-6)
            for name in grads:
                grads[name] *= scale

        # Update parameters
        self.optimizer.step(self.model.params, grads)

        # Update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Compute accuracy
        acc = accuracy(logits, targets, ignore_index=-1)

        return loss, acc

    def evaluate(self, max_steps: Optional[int] = None) -> Tuple[float, float, float]:
        """
        Evaluate on validation set.

        Args:
            max_steps: Maximum number of validation steps

        Returns:
            avg_loss: Average loss
            avg_ppl: Average perplexity
            avg_acc: Average accuracy
        """
        losses = []
        ppls = []
        accs = []

        val_iter = iter(self.val_loader)
        steps = 0

        try:
            while True:
                if max_steps is not None and steps >= max_steps:
                    break

                inputs, targets = next(val_iter)

                # Forward pass only
                logits, cache = self.model.forward(inputs)
                loss, _ = self.model.backward(targets, cache, ignore_index=-1)

                # Metrics
                ppl = perplexity(logits, targets, ignore_index=-1)
                acc = accuracy(logits, targets, ignore_index=-1)

                if not np.isnan(loss):
                    losses.append(loss)
                    ppls.append(ppl)
                    accs.append(acc)

                steps += 1

        except StopIteration:
            pass

        if len(losses) == 0:
            return float('nan'), float('nan'), float('nan')

        return np.mean(losses), np.mean(ppls), np.mean(accs)

    def train(self, max_steps: int):
        """
        Train for a specified number of steps.

        Args:
            max_steps: Maximum number of training steps
        """
        print(f"Starting training for {max_steps} steps...")
        print(f"  Model parameters: {self.model.get_num_params():,}")
        print(f"  Save directory: {self.save_dir}")
        print()

        start_time = time.time()
        train_iter = iter(self.train_loader)

        while self.step < max_steps:
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                # Should not happen with InfiniteDataLoader, but just in case
                train_iter = iter(self.train_loader)
                inputs, targets = next(train_iter)

            # Training step
            loss, acc = self.train_step(inputs, targets)
            self.train_losses.append(loss)
            self.step += 1

            # Logging
            if self.step % self.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.step / elapsed
                current_lr = self.lr_scheduler.get_lr() if self.lr_scheduler else self.optimizer.lr

                print(f"Step {self.step}/{max_steps} | "
                      f"Loss: {loss:.4f} | "
                      f"Acc: {acc:.3f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Steps/s: {steps_per_sec:.2f}")

            # Evaluation
            if self.step % self.eval_interval == 0:
                print(f"\n{'='*60}")
                print(f"Evaluation at step {self.step}")
                val_loss, val_ppl, val_acc = self.evaluate(max_steps=50)
                self.val_losses.append((self.step, val_loss))
                self.val_perplexities.append((self.step, val_ppl))

                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Perplexity: {val_ppl:.4f}")
                print(f"  Val Accuracy: {val_acc:.3f}")
                print(f"{'='*60}\n")

            # Save checkpoint
            if self.step % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.step}.npz")

        # Final evaluation and save
        print(f"\n{'='*60}")
        print("Final evaluation")
        val_loss, val_ppl, val_acc = self.evaluate()
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Perplexity: {val_ppl:.4f}")
        print(f"  Val Accuracy: {val_acc:.3f}")
        print(f"{'='*60}\n")

        self.save_checkpoint("checkpoint_final.npz")
        self.save_training_history()

        total_time = time.time() - start_time
        print(f"Training complete! Total time: {total_time/60:.2f} minutes")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        filepath = os.path.join(self.save_dir, filename)
        print(f"Saving checkpoint to {filepath}...")

        # Save model parameters
        np.savez(filepath, **self.model.params)

        print(f"  Saved!")

    def save_training_history(self):
        """Save training history (losses, metrics)."""
        history_file = os.path.join(self.save_dir, "training_history.json")

        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_perplexities': self.val_perplexities,
            'total_steps': self.step
        }

        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

        print(f"Saved training history to {history_file}")


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train NumPy Transformer on TinyStories')

    # Data arguments
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data (tokenized jsonl)')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation data (tokenized jsonl)')
    parser.add_argument('--tokenizer_type', type=str, default='tiktoken',
                        choices=['tiktoken', 'lecture2_bpe', 'char'],
                        help='Tokenizer type')
    parser.add_argument('--tokenizer_model', type=str, default=None,
                        help='Path to tokenizer model file (for lecture2_bpe)')

    # Model arguments
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
                        help='Do not tie input/output embeddings')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='Sequence length for training')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Maximum training steps')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Warmup steps')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['constant', 'linear_decay', 'cosine'],
                        help='Learning rate schedule')

    # Logging arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Steps between logging')
    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Steps between evaluation')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='Steps between checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    print("="*60)
    print("Training NumPy Transformer Language Model")
    print("="*60)
    print()

    # Load data
    print("Loading data...")
    train_dataset = TinyStoriesDataset(args.train_data, max_seq_len=args.max_seq_len)
    val_dataset = TinyStoriesDataset(args.val_data, max_seq_len=args.max_seq_len)

    train_loader = InfiniteDataLoader(train_dataset, args.batch_size, args.seq_len, shuffle=True)
    val_loader = InfiniteDataLoader(val_dataset, args.batch_size, args.seq_len, shuffle=False)

    print(f"  Train dataset: {len(train_dataset)} sequences")
    print(f"  Val dataset: {len(val_dataset)} sequences")
    print()

    # Create model
    print("Creating model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        use_learned_pos=args.use_learned_pos,
        tie_embeddings=not args.no_tie_embeddings
    )

    print(f"  Parameters: {model.get_num_params():,}")
    print()

    # Create optimizer
    print("Creating optimizer...")
    optimizer = AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Create LR scheduler
    lr_scheduler = LRScheduler(
        optimizer,
        schedule_type=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps,
        min_lr=args.learning_rate * 0.1
    )

    print(f"  Optimizer: AdamW")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  LR schedule: {args.lr_schedule}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print()

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval
    )

    # Train
    trainer.train(args.max_steps)


if __name__ == "__main__":
    main()