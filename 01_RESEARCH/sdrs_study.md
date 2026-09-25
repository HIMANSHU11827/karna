# Sparse Distributed Representations (SDRs) — Study
=====================================================

**Date:** 2026-09-23
**Author:** coder-3
**Purpose:** Research SDRs for the RT-ALM project

---

## 1. What Are SDRs?

A Sparse Distributed Representation is a binary vector x in {0,1}^D where:
- D is the dimensionality (typically 1,000-100,000)
- Exactly k bits are active (typically k = 0.02D, so 2% sparsity)
- The remaining D-k bits are zero

**Example:**
```
D = 20, k = 4
x = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
      ^              ^                    ^              ^
      bit 2          bit 7                bit 15         bit 19
```

The key idea: information is distributed across many bits, and only a small fraction are active at any time.

---

## 2. Properties of SDRs

### 2.1 Overlap

The overlap between two SDRs measures similarity:

```
overlap(x, y) = sum_i x_i * y_i
```

- overlap(x, x) = k (maximum similarity)
- overlap(x, y) = k * similarity (0 to k)
- overlap(x, y) = 0 means completely different

### 2.2 Union

The union combines two SDRs via bitwise OR:

```
union(x, y) = x OR y
```

- Preserves information from both
- Result has approximately 2k active bits (if x and y are sufficiently different)
- Can be used to compose representations

### 2.3 XOR Binding

XOR binding creates a new representation from two SDRs:

```
bind(x, y) = x XOR y
```

**Properties:**
- Inversible: unbind(bind(x, y), y) = x
- Distributive: bind(x, y) is similar to x if y is fixed
- Enables compositional semantics: bind(cat, black) = black_cat

### 2.4 Similarity

Similarity is normalized overlap:

```
sim(x, y) = overlap(x, y) / k
```

- sim(x, x) = 1.0 (identical)
- sim(x, y) = 0.0 (completely different)
- sim(x, y) > 0.5 typically means "similar enough"

---

## 3. Kanerva's Sparse Distributed Memory (SDM)

### 3.1 Architecture

Kanerva's SDM (1988) is a memory system based on high-dimensional binary vectors:

**Storage:**
- Patterns are stored as points in a D-dimensional binary space
- Each pattern is an address-value pair
- Writing: activate all addresses within radius r of the write address
- Reading: activate all addresses within radius r of the read address, average their values

**Key properties:**
- Robust to noise: partial cues retrieve full patterns
- Distributed: no single bit stores complete information
- Content-addressable: retrieve by content, not address

### 3.2 Capacity

For D-dimensional SDRs with k active bits:
- Capacity: approximately (D choose k) / (2k choose k) patterns
- For D=1000, k=20: ~10^30 patterns
- For D=10000, k=200: ~10^300 patterns

### 3.3 Noise Tolerance

SDRs can tolerate:
- Up to 50% bit flips and still retrieve correct pattern
- Up to 50% missing bits and still retrieve correct pattern
- Up to 50% extra bits and still retrieve correct pattern

---

## 4. Why SDRs Are Useful

### 4.1 Robustness

**Noise tolerance:** Even with 50% bit errors, SDRs can still be matched correctly.

**Graceful degradation:** As bits are removed, similarity decreases gradually rather than catastrophically.

**Biological plausibility:** The brain uses sparse firing patterns — only 1-5% of neurons fire at any time.

### 4.2 Compositionality

**XOR binding** allows creating new concepts from existing ones:
- bind(animal, striped) = striped_animal
- bind(animal, spotted) = spotted_animal
- bind(striped_animal, large) = large_striped_animal

**Unbinding** recovers original components:
- unbind(large_striped_animal, striped) = large_animal

### 4.3 Fast Similarity Search

**Hamming distance** between two SDRs:
```
hamming(x, y) = sum_i (x_i XOR y_i)
```

- Computationally cheap: O(D) with bitwise operations
- Can be computed in parallel on GPUs
- Enables fast nearest-neighbor search

### 4.4 Distributed Knowledge

**No single point of failure:** Each bit contributes a small amount to the overall representation.

**Reusable components:** The same SDR can be part of many different concepts.

**Incremental learning:** New patterns can be added without affecting existing ones.

---

## 5. SDRs in Language Models

### 5.1 Word Embeddings vs. SDRs

| Property | Word Embeddings | SDRs |
|----------|----------------|------|
| Representation | Dense float vectors | Sparse binary vectors |
| Similarity | Cosine similarity | Overlap / Hamming distance |
| Composition | Vector addition | XOR binding |
| Noise tolerance | Low | High |
| Capacity | Limited | Massive |
| Interpretability | Low | High |

### 5.2 Character n-gram SDRs

For text encoding:
1. Extract character n-grams (e.g., "hello" → "hel", "ell", "llo")
2. Hash each n-gram to a bit position
3. Activate those bits in the SDR

**Advantages:**
- Handles misspellings naturally ("helo" shares n-grams with "hello")
- No vocabulary needed
- Works for any language
- Incremental: can add new patterns online

### 5.3 Semantic Composition

For sentence encoding:
1. Encode each word as SDR
2. Bind word SDRs with position SDRs
3. Union all bound SDRs

```
sentence = "cat sits"
cat_sdr = encode("cat")
sits_sdr = encode("sits")
pos1 = encode("position_1")
pos2 = encode("position_2")
sentence_sdr = union(bind(cat_sdr, pos1), bind(sits_sdr, pos2))
```

---

## 6. SDRs in RT-ALM

### 6.1 Encoding Pipeline

```
Text → Character n-grams → Hash → SDR bits → Sparse binary vector
```

**Example:**
```
Input: "hello world"
n-grams (size=3): "hel", "ell", "llo", "lo ", "o w", " wo", "wor", "orl", "rld"
Hash each to bit position: 42, 156, 789, 234, 567, 890, 123, 456, 678
SDR: bits at positions [42, 123, 156, 234, 456, 567, 678, 789, 890] are 1
```

### 6.2 Memory Storage

**Episodic memory:** Store (input_SDR, output_SDR) pairs
- Key: input SDR (hashed from input text)
- Value: response SDR (hashed from response text)
- Retrieval: find key with highest overlap to query SDR

**Semantic memory:** Store concept SDRs
- Key: concept SDR
- Value: associated SDRs
- Retrieval: attractor dynamics (Hopfield)

### 6.3 Response Generation

1. Encode query → query_SDR
2. Retrieve similar stored SDR (by overlap)
3. Decode SDR → text (via n-gram lookup table)
4. If no good match, generate new response

---

## 7. Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| D (dimensionality) | 10,000-100,000 | Higher = better separation |
| k (active bits) | 0.02D | 2% sparsity optimal |
| n-gram size | 3-5 | Balance context vs. sparsity |
| Hash function | MurmurHash3 | Fast, good distribution |
| Similarity threshold | 0.5 | Minimum acceptable match |

---

## 8. References

1. **Kanerva (1988):** "Sparse Distributed Memory" — foundational work
2. **Maass (2000):** "On the Computational Power of Winner-Take-All" — k-WTA
3. **Rachkovskij & Kussul (2001):** "Binding and Normalization of Binary Sparse Distributed Representations"
4. **Kanerva (2009):** "Hyperdimensional Computing" — modern SDR applications
5. **Plate (1994):** "Distributed Representations" — Holographic Reduced Representations
---

*This study informs the SDR encoder design for RT-ALM.*
