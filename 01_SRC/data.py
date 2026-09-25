"""
Data loading for MNIST and CIFAR-10.

Loads raw binary data — no external dependencies beyond NumPy.
"""

import numpy as np
import struct
import os
from pathlib import Path
from typing import Tuple, Optional


def load_mnist(data_dir: str = "/home/himanshu/Desktop/AGI_RESEARCH_LAB/02_DATA") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset from raw binary files.
    
    Downloads files if not present. Returns normalized, flattened images.
    
    Returns:
        (x_train, y_train, x_test, y_test) where:
            x_train: (60000, 784) float64, values in [0, 1]
            y_train: (60000,) int64, digit labels
            x_test:  (10000, 784) float64
            y_test:  (10000,) int64
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    urls = {
        "train-images-idx3-ubyte": "https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte": "https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte": "https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte": "https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    }
    
    # Download if needed
    for name, url in urls.items():
        path = data_dir / name
        if not path.exists():
            import urllib.request
            import gzip
            print(f"Downloading {name}...")
            gz_path = data_dir / (name + ".gz")
            urllib.request.urlretrieve(url, gz_path)
            with gzip.open(gz_path, "rb") as f_in:
                with open(path, "wb") as f_out:
                    f_out.write(f_in.read())
            gz_path.unlink()
    
    def read_images(path: Path) -> np.ndarray:
        with open(path, "rb") as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
            return data.astype(np.float64) / 255.0
    
    def read_labels(path: Path) -> np.ndarray:
        with open(path, "rb") as f:
            magic, num = struct.unpack(">II", f.read(8))
            assert magic == 2049
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    
    x_train = read_images(data_dir / "train-images-idx3-ubyte")
    y_train = read_labels(data_dir / "train-labels-idx1-ubyte")
    x_test = read_images(data_dir / "t10k-images-idx3-ubyte")
    y_test = read_labels(data_dir / "t10k-labels-idx1-ubyte")
    
    return x_train, y_train, x_test, y_test


def one_hot(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Convert integer labels to one-hot vectors."""
    one_hot = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def create_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> list:
    """Create batches from data."""
    rng = rng or np.random.default_rng(42)
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    
    batches = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        batches.append((x[batch_idx], y[batch_idx]))
    
    return batches
