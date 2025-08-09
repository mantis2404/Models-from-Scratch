
# MLP (Multi-Layer Perceptron) from Scratch

This project implements a **Multi-Layer Perceptron (MLP)** using only `NumPy`, completely from scratch. It also compares the results with TensorFlow's  `MLP Sequential` model to validate correctness.

## Objectives

- Understand how neural networks work internally
- Build MLP components manually: forward pass, backpropagation, weight updates
- Implement training and evaluation loop
- Compare accuracy and training loss against a TensorFlow MLP

---

## Components and Functions Explained

### 1. **Imports and Dataset Preparation**

```python
X = np.random.rand(100,  10)
y = np.random.randint(0,  2, size=(100,  1))
```

- Made a dummy data for binary classification

---

### 2. **Activation Functions**

```python
def sigmoid(x): ...
def sigmoid_derivative(x): ...
def relu(x):.....
def relu_derivative(x):...
```

- **Sigmoid** maps real values to (0,1).
- Its **derivative** is used for backpropagation.
- **ReLU** is used for hidden layers
- Its derivative is used for backpropagation fo hidden layers.

---

### 3. **Softmax and Cross-Entropy Loss**

```python
def binary_cross_entropy(y_true, y_pred_prob): ...
def binary_croos_entropy_derivative(y_true,y_pred_prob):...
```

- **binary_cross-entropy Loss** to compute difference between true and predicted distribution.
- **binary_cross_entropy_derivative** to help in backpropagation

---

### 4. **Weight Initialization**

```python
def glorot_unifrom(fan_in, fan_out): ...
def glorot_normal(fan_in, fan_out): ...

def he_unifrom(fan_in, fan_out): ...
def he_normal(fan_in, fan_out): ...
```

- Initializes `W1`, `b1`, `W2`, `b2` with small random numbers for the two layers.
- **glorot** for tanh and sigmoid
- **he** for relu

---

### 5. **Forward Pass**



---

### 6. **Backpropagation**


---

### 7. **Parameter Update**

---

### 8. **Model Training**

```python
def  mlp(X,y,batch_size,epochs,layers,validation_split):....
```
- `validation_split`: the fraction of X to be used as testing data
- `batch_size`: number of samples to be sent in one batch
- `layers`: number of hidden layers
- Iteratively:
  - Forward pass
  - Compute loss
  - Backward pass
  - Update weights
  - Print accuracy and loss

---

### 9. **Prediction**

```python
mlp(X,y,32,100,3,0.2)....
```

- Returns the accuracy and loss for each epoch

---

## Comparison with TensorFlow MLP

```python
model = Sequential(...)
```

- A TensorFlow equivalent model is trained using the same dataset for benchmarking.
- Both accuracy and loss curves are plotted and compared.

---

## Results

| Model             | Final Accuracy |
|------------------|----------------|
| MLP from Scratch | ~58%           |
| TensorFlow MLP      | ~54%           |

The similarity in results confirms correctness of implementation.

---

## Visualizations

- Loss vs Epoch
- Accuracy vs Epoch
- Comparison of Scratch vs TensorFlow models

---

## Conclusion

This project shows how **neural networks** learn by implementing all fundamental operations from the ground up. The match with TensorFlow proves the math and code are correct. It builds deep intuition into how backpropagation and weight updates work.

