import sys
sys.path.insert(0, '.')
from raie_network import RAIENetwork, KWTA, SRBI, ComplementaryStore, XORMemory, IACMemory
import numpy as np

print("=== RAIE Network Smoke Test ===")

# Test KWTA
print("\n1. KWTA activation:")
x = np.random.randn(10)
y = KWTA.activate(x, 3)
print(f"   Input shape: {x.shape}, Output shape: {y.shape}")
print(f"   Sparsity: {1.0 - np.count_nonzero(y) / len(y):.2f}")

# Test SRBI
print("\n2. SRBI initialization:")
W = SRBI.init_weights(10, 20, 0.1)
print(f"   Weight shape: {W.shape}")
print(f"   Sparsity: {1.0 - np.count_nonzero(W) / W.size:.2f}")

# Test ComplementaryStore
print("\n3. Complementary Learning:")
cs = ComplementaryStore(20, 10)
x = np.random.randn(20)
y = np.random.randn(10)
cs.store_fast(x, y)
cs.store_slow(x, y)
pred = cs.predict(x)
print(f"   Prediction shape: {pred.shape}")

# Test XORMemory
print("\n4. XOR Binding:")
xor = XORMemory(10, 0.1)
a = np.random.randn(10)
b = np.random.randn(10)
bound = xor.bind(a, b)
print(f"   Bound shape: {bound.shape}")

# Test IACMemory
print("\n5. IAC Memory:")
iac = IACMemory(10, 5)
pattern = np.random.randn(10)
idx = iac.store(pattern, 3)
retrieved_label = iac.predict_label(pattern)
print(f"   Stored at: {idx}, Retrieved label: {retrieved_label}")

# Test RAIELayer
print("\n6. RAIE Layer:")
from raie_network import RAIELayer
layer = RAIELayer(20, 10)
x = np.random.randn(20)
y = layer.forward(x)
print(f"   Input: {x.shape}, Output: {y.shape}")

# Test RAIENetwork
print("\n7. RAIE Network:")
net = RAIENetwork([784, 64, 10])
x = np.random.randn(784)
activations = net.forward(x)
print(f"   Activations: {[a.shape for a in activations]}")

# Test training
print("\n8. Training step:")
net.train_step(x, 5)
print("   Training step completed")

# Test prediction
print("\n9. Prediction:")
label, output = net.predict(x)
print(f"   Predicted label: {label}, Output shape: {output.shape}")

print("\n=== All tests passed! ===")
