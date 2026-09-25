import sys
sys.path.insert(0, '../19_PROTOTYPES')
from spmn_network import SPMNetwork
import numpy as np

# Quick smoke test
net = SPMNetwork([784, 64, 10], learning_rate=0.01, sparsity_threshold=0.9)
print('Network created:', [l.W.shape for l in net.layers])

# Forward pass test
x = np.random.randn(784)
activations = net.forward(x)
print('Activations:', [a.shape for a in activations])

# Inference test
pred = net.predict(x)
print('Prediction shape:', pred.shape)
print('Prediction:', pred[:5])

# Stats
stats = net.get_all_stats()
print('Layer stats:', stats)
print('Network sparsity:', net.total_sparsity())
print('OK')
