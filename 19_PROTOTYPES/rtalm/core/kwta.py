"""
RT-ALM: k-Winners-Take-All (k-WTA) Activation
================================================
Sparse activation function that selects top-k neurons.

Based on: Maass (2000) — "On the computational power of circuits of spiking neurons"
"""
import numpy as np


def kwta_activate(activations: np.ndarray, k: int) -> np.ndarray:
    """
    k-Winners-Take-All activation.
    
    Selects top-k neurons, sets rest to 0.
    If k >= len(activations), returns all non-negative.
    
    Args:
        activations: Raw activation values (1D array)
        k: Number of winners to select
    
    Returns:
        Sparse binary array with exactly k active elements (or fewer if ties)
    """
    n = len(activations)
    if k <= 0:
        return np.zeros(n, dtype=np.int8)
    if k >= n:
        return (activations >= 0).astype(np.int8)
    
    # Find the k-th largest value as threshold
    # Using argpartition for O(n) instead of O(n log n)
    top_k_indices = np.argpartition(activations, -k)[-k:]
    threshold = activations[top_k_indices].min()
    
    # Create sparse output
    sparse = np.zeros(n, dtype=np.int8)
    sparse[activations >= threshold] = 1
    
    # Handle ties: if more than k are above threshold, keep only top-k
    active_count = sparse.sum()
    if active_count > k:
        # Among those above threshold, keep only top-k by value
        above_threshold = np.where(activations >= threshold)[0]
        top_k = above_threshold[np.argpartition(activations[above_threshold], -k)[-k:]]
        sparse[:] = 0
        sparse[top_k] = 1
    
    return sparse


def kwta_activate_continuous(activations: np.ndarray, k: int) -> np.ndarray:
    """
    Continuous k-WTA: returns raw values for top-k, 0 for rest.
    Useful for gradient-based learning (though we don't use backprop here).
    """
    n = len(activations)
    if k <= 0:
        return np.zeros(n, dtype=np.float32)
    if k >= n:
        return activations.copy()
    
    top_k_indices = np.argpartition(activations, -k)[-k:]
    threshold = activations[top_k_indices].min()
    
    result = np.zeros(n, dtype=np.float32)
    mask = activations >= threshold
    result[mask] = activations[mask]
    
    # Handle ties
    active_count = mask.sum()
    if active_count > k:
        above_threshold = np.where(mask)[0]
        top_k = above_threshold[np.argpartition(activations[above_threshold], -k)[-k:]]
        result[:] = 0
        result[top_k] = activations[top_k]
    
    return result


def sparsity_level(n: int, k: int) -> float:
    """Returns the sparsity level (fraction of active neurons)."""
    return k / n if n > 0 else 0.0


if __name__ == "__main__":
    # Test k-WTA
    np.random.seed(42)
    x = np.random.randn(100)
    
    for k in [1, 5, 10, 20, 50]:
        sparse = kwta_activate(x, k)
        print(f"k={k:3d}: active={sparse.sum():3d}, sparsity={sparsity_level(100, k):.2%}")
        assert sparse.sum() == min(k, 100), f"Expected {k} active, got {sparse.sum()}"
    
    print("\nAll k-WTA tests passed!")
