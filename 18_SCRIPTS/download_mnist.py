#!/usr/bin/env python3
"""Download MNIST dataset."""
import urllib.request
import gzip
import os
from pathlib import Path

data_dir = Path("/home/himanshu/Desktop/AGI_RESEARCH_LAB/16_DATA")
data_dir.mkdir(parents=True, exist_ok=True)

base_url = "https://yann.lecun.com/exdb/mnist/"
files = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

for name, filename in files.items():
    filepath = data_dir / filename
    if not filepath.exists():
        url = base_url + filename
        print(f"Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  Saved to {filepath}")
        except Exception as e:
            print(f"  Failed: {e}")
            # Try alternative URL
            alt_url = f"https://ossci-datasets.s3.amazonaws.com/mnist/{filename}"
            print(f"  Trying alternative: {alt_url}")
            try:
                urllib.request.urlretrieve(alt_url, filepath)
                print(f"  Saved to {filepath}")
            except Exception as e2:
                print(f"  Also failed: {e2}")

print("\nDone.")
