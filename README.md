
# Machine Learning from Scratch – MLP and t-SNE

This repository contains two independent, from-scratch implementations of fundamental machine learning techniques:

1. **Multi-Layer Perceptron (MLP)** - a feedforward neural network for classification tasks.
2. **t-Distributed Stochastic Neighbor Embedding (t-SNE)** - a nonlinear dimensionality reduction technique for visualization of high-dimensional data.

Each implementation is contained in its own folder with a dedicated Jupyter Notebook (`.ipynb`) and a detailed README explaining the algorithm, step-by-step code, and comparisons with established libraries.

---

## Repository Structure

```
.
├── MLP/
│   ├── MLP.ipynb
│   ├── README.md
├── T-SNE/
│   ├── T-SNE.ipynb
│   ├── README.md
└── README.md
```

---

## 1. MLP (Multi-Layer Perceptron)

**Goal:** Implement a fully-connected feedforward neural network using only NumPy, from scratch.

**Highlights:**
- Forward propagation and backpropagation implemented manually.
- Binary cross-entropy loss function and ReLU/Sigmoid activations.
- Custom weight initialization (Glorot and He methods).
- Training loop with mini-batch gradient descent.
- Benchmarking against TensorFlow's Sequential MLP.

**Results:** Comparable accuracy and loss trends with TensorFlow, validating the correctness of the implementation.

For detailed explanation, see [`MLP/README.md`](MLP/MLP%20(MULTI-LAYER%20PERCEPTRON)%20FROM%20SCRATCH.md.md).

---

## 2. t-SNE

**Goal:** Implement t-SNE from scratch to project high-dimensional datasets into 2D while preserving local structure.

**Highlights:**
- Pairwise similarity computation in high-dimensional space.
- Perplexity-based adaptive variance selection using binary search.
- Low-dimensional embedding updates with gradient descent, momentum, and gains.
- Early exaggeration to improve cluster separation.
- Comparison with scikit-learn's t-SNE in terms of output and runtime.

**Results:** Produces visually similar cluster patterns to sklearn's t-SNE, though significantly slower due to pure NumPy implementation.

For detailed explanation, see [`T-SNE/README.md`](T-SNE/t-sne_readme.md).

---

## Requirements

- Python 3.8+
- Jupyter Notebook
- Libraries:
  ```
  numpy
  pandas
  matplotlib
  seaborn  # optional, for visualization
  scikit-learn  # only for dataset loading or comparison
  tensorflow  # optional, for MLP benchmarking
  ```

---

## Running the Notebooks

The notebooks have integrated google colab links- you can view and run them.
