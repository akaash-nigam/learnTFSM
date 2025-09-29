"""
Optimizers for training the language model.
"""
import numpy as np
from typing import Dict


class Optimizer:
    """Base optimizer class."""

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """Update parameters given gradients."""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize SGD optimizer.

        Args:
            learning_rate: Learning rate for parameter updates
            momentum: Momentum coefficient (0 for no momentum)
        """
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """
        Update parameters using SGD with optional momentum.

        Args:
            params: Dictionary of parameters to update
            grads: Dictionary of gradients
        """
        for name, param in params.items():
            if name not in grads:
                continue  # Skip parameters without gradients (e.g., fixed pos encoding)

            grad = grads[name]

            if self.momentum > 0:
                # Initialize velocity if needed
                if name not in self.velocity:
                    self.velocity[name] = np.zeros_like(param)

                # Update velocity: v = momentum * v - lr * grad
                self.velocity[name] = self.momentum * self.velocity[name] - self.lr * grad

                # Update parameter: param += v
                param += self.velocity[name]
            else:
                # Simple SGD: param -= lr * grad
                param -= self.lr * grad


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).

    Reference: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.0):
        """
        Initialize Adam optimizer.

        Args:
            learning_rate: Learning rate (alpha)
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            weight_decay: L2 regularization coefficient
        """
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        # State variables
        self.m = {}  # First moment estimates (mean)
        self.v = {}  # Second moment estimates (variance)
        self.t = 0   # Time step

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """
        Update parameters using Adam optimizer.

        Args:
            params: Dictionary of parameters to update
            grads: Dictionary of gradients
        """
        self.t += 1

        for name, param in params.items():
            if name not in grads:
                continue  # Skip parameters without gradients

            grad = grads[name]

            # Add weight decay if specified
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            # Initialize moment estimates if needed
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)

            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Update parameters
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def reset(self):
        """Reset optimizer state (useful for starting new training run)."""
        self.m = {}
        self.v = {}
        self.t = 0


class AdamW(Optimizer):
    """
    AdamW optimizer (Adam with decoupled weight decay).

    Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)

    AdamW decouples weight decay from gradient-based updates, which often
    leads to better generalization than standard Adam with L2 regularization.
    """

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 weight_decay: float = 0.01):
        """
        Initialize AdamW optimizer.

        Args:
            learning_rate: Learning rate (alpha)
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            weight_decay: Decoupled weight decay coefficient
        """
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        # State variables
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Time step

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """
        Update parameters using AdamW optimizer.

        Args:
            params: Dictionary of parameters to update
            grads: Dictionary of gradients
        """
        self.t += 1

        for name, param in params.items():
            if name not in grads:
                continue

            grad = grads[name]

            # Initialize moment estimates if needed
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)

            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad

            # Update biased second raw moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)

            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Update parameters with Adam step
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Apply decoupled weight decay
            if self.weight_decay > 0:
                param -= self.lr * self.weight_decay * param

    def reset(self):
        """Reset optimizer state."""
        self.m = {}
        self.v = {}
        self.t = 0


class LRScheduler:
    """Learning rate scheduler."""

    def __init__(self, optimizer: Optimizer, schedule_type: str = 'constant',
                 warmup_steps: int = 0, total_steps: int = None,
                 min_lr: float = 0.0):
        """
        Initialize learning rate scheduler.

        Args:
            optimizer: Optimizer to schedule
            schedule_type: Type of schedule ('constant', 'linear_decay', 'cosine')
            warmup_steps: Number of warmup steps
            total_steps: Total training steps (required for decay schedules)
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.current_step = 0

    def step(self):
        """Update learning rate based on schedule."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        elif self.schedule_type == 'constant':
            lr = self.base_lr
        elif self.schedule_type == 'linear_decay':
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr + (self.min_lr - self.base_lr) * progress
        elif self.schedule_type == 'cosine':
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        self.optimizer.lr = max(lr, self.min_lr)

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr