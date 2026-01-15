"""Tests for the PPO trainer module."""

import pytest


def test_ppo_import():
    """Test that SafePPOTrainer can be imported."""
    try:
        import safe
        assert hasattr(safe, 'SafePPO'), "SafePPO should be available in safe module"
        assert hasattr(safe, 'SafePPOTrainer') is False, "SafePPOTrainer should not be directly exposed"

        from safe.ppo import SafePPOTrainer, SafePPO
        assert SafePPOTrainer is not None
        assert SafePPO is SafePPOTrainer  # Check alias
    except ImportError as e:
        pytest.skip(f"Cannot test PPO import due to missing dependencies: {e}")


def test_ppo_class_exists():
    """Test that SafePPOTrainer class has required methods."""
    try:
        from safe.ppo import SafePPOTrainer

        # Check that the class has the required methods
        assert hasattr(SafePPOTrainer, '__init__')
        assert hasattr(SafePPOTrainer, 'train')
        assert hasattr(SafePPOTrainer, '_generate_with_log_probs')
        assert hasattr(SafePPOTrainer, '_calculate_rewards')
        assert hasattr(SafePPOTrainer, '_calculate_kl_divergence')
        assert hasattr(SafePPOTrainer, '_compute_ppo_loss')
        assert hasattr(SafePPOTrainer, '_save_model')
    except ImportError as e:
        pytest.skip(f"Cannot test PPO class due to missing dependencies: {e}")


def test_ppo_supported_methods():
    """Test that all expected generation methods are supported."""
    try:
        from safe.ppo import SafePPOTrainer

        # The supported methods should be in the train method
        # We can't easily test this without actually running the trainer,
        # but we can check that the class definition is correct
        import inspect
        train_method = inspect.getsource(SafePPOTrainer.train)

        # Check that all expected methods are mentioned
        expected_methods = [
            'de_novo_generation',
            'motif_extension',
            'linker_generation',
            'scaffold_decoration',
            'super_structure',
        ]

        for method in expected_methods:
            assert method in train_method, f"Method '{method}' should be supported in train()"

    except ImportError as e:
        pytest.skip(f"Cannot test PPO supported methods due to missing dependencies: {e}")
