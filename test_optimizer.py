"""
Test optimizer implementations.
"""
import numpy as np
from optimizer import SGD, Adam, AdamW, LRScheduler

np.random.seed(42)


def test_sgd_basic():
    """Test basic SGD without momentum."""
    print("Testing basic SGD...")

    # Create simple parameters
    params = {
        'W': np.array([[1.0, 2.0], [3.0, 4.0]]),
        'b': np.array([0.5, 1.5])
    }

    grads = {
        'W': np.array([[0.1, 0.2], [0.3, 0.4]]),
        'b': np.array([0.1, 0.2])
    }

    # Store original values
    W_orig = params['W'].copy()
    b_orig = params['b'].copy()

    # Update with SGD
    optimizer = SGD(learning_rate=0.1, momentum=0.0)
    optimizer.step(params, grads)

    # Check updates
    expected_W = W_orig - 0.1 * grads['W']
    expected_b = b_orig - 0.1 * grads['b']

    assert np.allclose(params['W'], expected_W), "SGD weight update incorrect"
    assert np.allclose(params['b'], expected_b), "SGD bias update incorrect"

    print(f"  W change: {np.abs(params['W'] - W_orig).max():.6f}")
    print(f"  b change: {np.abs(params['b'] - b_orig).max():.6f}")
    print("  ✅ Basic SGD passed!\n")


def test_sgd_momentum():
    """Test SGD with momentum."""
    print("Testing SGD with momentum...")

    params = {
        'W': np.array([[1.0, 2.0], [3.0, 4.0]])
    }

    grads1 = {
        'W': np.array([[0.1, 0.2], [0.3, 0.4]])
    }

    grads2 = {
        'W': np.array([[0.2, 0.1], [0.4, 0.3]])
    }

    optimizer = SGD(learning_rate=0.1, momentum=0.9)

    W_orig = params['W'].copy()

    # First step
    optimizer.step(params, grads1)
    W_after_step1 = params['W'].copy()

    # Second step
    optimizer.step(params, grads2)
    W_after_step2 = params['W'].copy()

    # Manually compute expected values
    v1 = -0.1 * grads1['W']
    W1 = W_orig + v1

    v2 = 0.9 * v1 - 0.1 * grads2['W']
    W2 = W1 + v2

    assert np.allclose(W_after_step1, W1), "First momentum step incorrect"
    assert np.allclose(W_after_step2, W2), "Second momentum step incorrect"

    print(f"  Step 1 change: {np.abs(W_after_step1 - W_orig).max():.6f}")
    print(f"  Step 2 change: {np.abs(W_after_step2 - W_after_step1).max():.6f}")
    print("  ✅ SGD with momentum passed!\n")


def test_adam_basic():
    """Test Adam optimizer."""
    print("Testing Adam optimizer...")

    params = {
        'W': np.array([[1.0, 2.0], [3.0, 4.0]])
    }

    grads = {
        'W': np.array([[0.1, 0.2], [0.3, 0.4]])
    }

    optimizer = Adam(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

    W_orig = params['W'].copy()

    # First step
    optimizer.step(params, grads)

    # Manually compute expected update
    m = (1 - 0.9) * grads['W']
    v = (1 - 0.999) * (grads['W'] ** 2)

    m_hat = m / (1 - 0.9 ** 1)
    v_hat = v / (1 - 0.999 ** 1)

    expected_W = W_orig - 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)

    assert np.allclose(params['W'], expected_W, rtol=1e-6), "Adam update incorrect"
    assert optimizer.t == 1, "Time step not incremented"

    print(f"  W change: {np.abs(params['W'] - W_orig).max():.6f}")
    print(f"  Time step: {optimizer.t}")
    print("  ✅ Adam optimizer passed!\n")


def test_adam_multiple_steps():
    """Test Adam over multiple steps."""
    print("Testing Adam over multiple steps...")

    params = {
        'W': np.array([[1.0, 2.0]])
    }

    optimizer = Adam(learning_rate=0.01, beta1=0.9, beta2=0.999)

    losses = []
    for i in range(10):
        # Simulate gradient that points toward zero
        grads = {'W': params['W'].copy()}
        optimizer.step(params, grads)
        loss = (params['W'] ** 2).sum()
        losses.append(loss)

    # Loss should decrease
    assert losses[-1] < losses[0], "Loss should decrease over time"
    assert optimizer.t == 10, "Time step should be 10"

    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Time step: {optimizer.t}")
    print("  ✅ Adam multiple steps passed!\n")


def test_adam_weight_decay():
    """Test Adam with weight decay."""
    print("Testing Adam with weight decay...")

    params = {
        'W': np.array([[1.0, 2.0]])
    }

    grads = {
        'W': np.array([[0.1, 0.1]])
    }

    # Adam with weight decay
    optimizer_wd = Adam(learning_rate=0.01, weight_decay=0.01)
    params_wd = {'W': params['W'].copy()}

    # Adam without weight decay
    optimizer_no_wd = Adam(learning_rate=0.01, weight_decay=0.0)
    params_no_wd = {'W': params['W'].copy()}

    optimizer_wd.step(params_wd, grads)
    optimizer_no_wd.step(params_no_wd, grads)

    # With weight decay should pull params closer to zero
    assert np.abs(params_wd['W']).sum() < np.abs(params_no_wd['W']).sum(), \
        "Weight decay should shrink parameters more"

    print(f"  Without WD: {params_no_wd['W']}")
    print(f"  With WD: {params_wd['W']}")
    print("  ✅ Adam weight decay passed!\n")


def test_adamw():
    """Test AdamW optimizer (decoupled weight decay)."""
    print("Testing AdamW optimizer...")

    params = {
        'W': np.array([[1.0, 2.0]])
    }

    grads = {
        'W': np.array([[0.1, 0.1]])
    }

    optimizer = AdamW(learning_rate=0.01, weight_decay=0.01)

    W_orig = params['W'].copy()
    optimizer.step(params, grads)

    # Weight should have changed
    assert not np.allclose(params['W'], W_orig), "Parameters should change"
    assert np.abs(params['W']).sum() < np.abs(W_orig).sum(), \
        "Decoupled weight decay should shrink parameters"

    print(f"  Original W: {W_orig}")
    print(f"  Updated W: {params['W']}")
    print("  ✅ AdamW passed!\n")


def test_lr_scheduler_warmup():
    """Test learning rate scheduler with warmup."""
    print("Testing LR scheduler with warmup...")

    params = {'W': np.array([[1.0]])}
    optimizer = Adam(learning_rate=0.1)
    scheduler = LRScheduler(optimizer, schedule_type='constant', warmup_steps=5)

    base_lr = 0.1

    # During warmup
    for step in range(1, 6):
        scheduler.step()
        expected_lr = base_lr * (step / 5)
        assert np.isclose(optimizer.lr, expected_lr), \
            f"Warmup LR incorrect at step {step}"

    # After warmup
    scheduler.step()
    assert np.isclose(optimizer.lr, base_lr), "LR after warmup should be base_lr"

    print(f"  Base LR: {base_lr}")
    print(f"  LR at step 3: {base_lr * 0.6:.4f}")
    print(f"  LR at step 6: {optimizer.lr:.4f}")
    print("  ✅ LR scheduler warmup passed!\n")


def test_lr_scheduler_linear_decay():
    """Test learning rate scheduler with linear decay."""
    print("Testing LR scheduler with linear decay...")

    params = {'W': np.array([[1.0]])}
    optimizer = Adam(learning_rate=1.0)
    scheduler = LRScheduler(
        optimizer,
        schedule_type='linear_decay',
        warmup_steps=0,
        total_steps=10,
        min_lr=0.0
    )

    lrs = []
    for _ in range(10):
        scheduler.step()
        lrs.append(optimizer.lr)

    # Should decay linearly from 1.0 to 0.0
    assert lrs[0] > lrs[-1], "LR should decrease"
    assert np.isclose(lrs[-1], 0.0, atol=1e-6), "Final LR should be min_lr"

    print(f"  Initial LR: {lrs[0]:.4f}")
    print(f"  Final LR: {lrs[-1]:.4f}")
    print("  ✅ LR scheduler linear decay passed!\n")


def test_lr_scheduler_cosine():
    """Test learning rate scheduler with cosine decay."""
    print("Testing LR scheduler with cosine decay...")

    params = {'W': np.array([[1.0]])}
    optimizer = Adam(learning_rate=1.0)
    scheduler = LRScheduler(
        optimizer,
        schedule_type='cosine',
        warmup_steps=0,
        total_steps=100,
        min_lr=0.0
    )

    lrs = []
    for _ in range(100):
        scheduler.step()
        lrs.append(optimizer.lr)

    # Should decay from 1.0 to 0.0 following cosine curve
    assert lrs[0] > lrs[49] > lrs[-1], "LR should decrease monotonically"
    assert np.isclose(lrs[-1], 0.0, atol=1e-6), "Final LR should be min_lr"

    # Cosine should decay slower initially than linear
    print(f"  Initial LR: {lrs[0]:.4f}")
    print(f"  Mid LR: {lrs[49]:.4f}")
    print(f"  Final LR: {lrs[-1]:.4f}")
    print("  ✅ LR scheduler cosine passed!\n")


def test_skip_params_without_grads():
    """Test that optimizers skip parameters without gradients."""
    print("Testing skip parameters without gradients...")

    params = {
        'W': np.array([[1.0, 2.0]]),
        'pos_encoding': np.array([[0.1, 0.2]])  # No gradient
    }

    grads = {
        'W': np.array([[0.1, 0.2]])
        # 'pos_encoding' intentionally missing
    }

    pos_encoding_orig = params['pos_encoding'].copy()

    optimizer = Adam(learning_rate=0.01)
    optimizer.step(params, grads)

    # pos_encoding should not change
    assert np.allclose(params['pos_encoding'], pos_encoding_orig), \
        "Parameters without gradients should not change"

    print(f"  pos_encoding unchanged: {np.allclose(params['pos_encoding'], pos_encoding_orig)}")
    print("  ✅ Skip parameters test passed!\n")


def test_optimizer_reset():
    """Test optimizer state reset."""
    print("Testing optimizer reset...")

    params = {'W': np.array([[1.0, 2.0]])}
    grads = {'W': np.array([[0.1, 0.2]])}

    optimizer = Adam(learning_rate=0.01)

    # Take a few steps
    for _ in range(5):
        optimizer.step(params, grads)

    assert optimizer.t == 5, "Time step should be 5"
    assert len(optimizer.m) > 0, "Should have moment estimates"

    # Reset
    optimizer.reset()

    assert optimizer.t == 0, "Time step should be reset to 0"
    assert len(optimizer.m) == 0, "Moment estimates should be cleared"
    assert len(optimizer.v) == 0, "Second moments should be cleared"

    print(f"  Time step after reset: {optimizer.t}")
    print(f"  Moment dict size: {len(optimizer.m)}")
    print("  ✅ Optimizer reset passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Optimizers")
    print("=" * 60)
    print()

    test_sgd_basic()
    test_sgd_momentum()
    test_adam_basic()
    test_adam_multiple_steps()
    test_adam_weight_decay()
    test_adamw()
    test_lr_scheduler_warmup()
    test_lr_scheduler_linear_decay()
    test_lr_scheduler_cosine()
    test_skip_params_without_grads()
    test_optimizer_reset()

    print("=" * 60)
    print("🎉 All optimizer tests passed!")
    print("=" * 60)