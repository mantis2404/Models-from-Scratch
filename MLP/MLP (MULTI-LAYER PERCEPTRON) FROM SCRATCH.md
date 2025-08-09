
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

The forward pass computes the output of the network given an input batch. It is implemented using vectorized matrix operations for efficiency and clarity. For each layer the computation follows:

- Compute pre-activation: `Z = A_prev @ W + b`  
- Apply activation: `A = activation(Z)` (ReLU for hidden layers, Sigmoid for the final layer in this binary setup)
- Store intermediate values `(A_prev, Z, W, b)` needed later for backpropagation

A minimal pseudocode for a two-hidden-layer network:

```python
# X: shape (batch_size, n_features)
A0 = X

# Layer 1
Z1 = A0.dot(W1) + b1          # shape (batch_size, n_hidden1)
A1 = relu(Z1)

# Layer 2
Z2 = A1.dot(W2) + b2          # shape (batch_size, n_hidden2)
A2 = relu(Z2)

# Output layer
Z3 = A2.dot(W3) + b3          # shape (batch_size, 1)
A3 = sigmoid(Z3)              # predictions in (0,1)
```

Notes:
- Shapes must align: if `W1` is `(n_features, n_hidden1)`, then `A0.dot(W1)` yields `(batch_size, n_hidden1)`.
- Using `batch_size` allows mini-batch gradient descent.
- Keep the pre-activation `Z` values because activation derivatives are computed from them during backpropagation.

---

### 6. **Backpropagation**

Backpropagation computes gradients of the loss with respect to every parameter by moving backwards through the network using the chain rule. For binary cross-entropy loss with a sigmoid output the derivative simplifies cleanly, which we use in the implementation.

Key steps (vectorized):

1. **Output layer gradient**  
   For binary cross-entropy loss `L` and sigmoid activation `a = sigmoid(z)`:
   - `dA3 = - (y / a) + (1 - y) / (1 - a)`  (general form)
   - This simplifies to `dZ3 = A3 - y` for BCE paired with sigmoid (numerically stable and commonly used)
   - `dW3 = A2.T.dot(dZ3) / batch_size`
   - `db3 = np.sum(dZ3, axis=0, keepdims=True) / batch_size`

2. **Hidden layers**  
   Propagate the gradient backward through each hidden layer `l`:
   - `dA_l = dZ_{l+1}.dot(W_{l+1}.T)`
   - `dZ_l = dA_l * activation_derivative(Z_l)`  (for ReLU, derivative is 0 for Z<=0, 1 for Z>0)
   - `dW_l = A_{l-1}.T.dot(dZ_l) / batch_size`
   - `db_l = np.sum(dZ_l, axis=0, keepdims=True) / batch_size`

3. **Vectorized implementation**  
   All operations are performed with NumPy matrix ops to avoid Python loops and keep the code clean and fast. A typical backprop loop over layers (from last to first) looks like:

```python
dZ = A_output - y                     # for output layer
dW = A_prev.T.dot(dZ) / batch_size
db = np.sum(dZ, axis=0, keepdims=True) / batch_size

# iterate for each hidden layer in reverse
for l in reversed(range(1, L)):
    dA = dZ.dot(W_next.T)
    dZ = dA * relu_derivative(Z_l)
    dW = A_{l-1}.T.dot(dZ) / batch_size
    db = np.sum(dZ, axis=0, keepdims=True) / batch_size
```

Notes:
- Storing intermediate `A` and `Z` values during forward pass is essential for computing these derivatives.
- Gradient shapes:
  - `dW_l` has same shape as `W_l`
  - `db_l` has same shape as `b_l`
- If using numerical stability tricks (e.g., clipping `A` before log in BCE), apply them consistently to avoid NaNs.

---

### 7. **Parameter Update**

After computing gradients, parameters are updated using gradient descent (or its variants). The simplest update rule is vanilla stochastic gradient descent (SGD):

```python
W = W - learning_rate * dW
b = b - learning_rate * db
```

Implementation notes and common improvements:

- **Mini-batch updates:** Apply the update after computing gradients for a mini-batch. This balances gradient quality and compute efficiency.
- **Learning rate:** Choose a sensible `learning_rate` (e.g., `1e-3` or `1e-2`) and consider decaying it over epochs if training plateaus.
- **Momentum:** Optionally maintain a velocity term `v` for each parameter:
  ```python
  v = momentum * v - learning_rate * dW
  W += v
  ```
  Momentum helps accelerate convergence and smooth out noisy updates.
- **L2 regularization (weight decay):** If used, add `lambda * W` to the gradient before updating:
  ```python
  dW += l2_lambda * W
  W -= learning_rate * dW
  ```
- **Gradient clipping:** To avoid exploding gradients, clip `dW` to a maximum norm.
- **Initialization matching:** Ensure your chosen initialization (Glorot / He) matches activation functions to keep gradients stable.

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
