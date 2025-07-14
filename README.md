
# t-SNE from Scratch

This projects implements **t-SNE (t-distributed Stochastic Neighbor Embedding)**, an unsupervised learning algorithm using `NumPy` completely from scratch and compares it with sklearn's TSNE.
t-sne reduces high-dimensional data into low dimensions  for visualization while preserving local structure.

## Objectives

- Understand how t-sne work internally
- Build t-sne components manually: low and high dimension matrices, finding the optimal sigma
- Implement training and evaluation loop
- Compare running time and cluster formation against the sklearn TSNE

---
## Overview

### t-SNE works by:

- Computing pairwise similarities in high-dimensional space.
- Defining pairwise similarities in low-dimensional space.
- Iteratively minimizing the KL divergence between these distributions using gradient descent with momentum and gains.

---
## Components and Functions Explained
### 1. **Imports and Dataset Preparation**

```python
from sklearn.datasets import load_digits
digits=load_digits()
X=digits.data
target=digits.target
X=X/16
X.shape
```
```
(1797,64)
```

- Imported the **load_digits** dataset from sklearn
- normalized the input data

---

### 2. **Defining Distance Matrix**

``` python
dist_matrix(X):...
```

- Computes the pairwise squared **Euclidean distances** between all samples in `X` and low dimension embeddings(`y`).
- Returns a **n×n** matrix D where $D[i, j]$ is the squared distance between samples i and j where n is the number of samples in X

  
$$D_{ij} = \|x_i - x_j\|^2 $$

---

### 3. **Defining Probability Matrix For High Dimension**

``` python
probability_matrix(sigma, i):...
```

- Calculates the **conditional probability distribution** matrix P for higher dimensions given a sample (index i for X) and variance ($\sigma$).
- For point i, computes $p_{j|i}$ using a **Gaussian kernel**:
- it is asymmetric as variance is different for i and j
 - $p_{j|i}$ is row wise normalized
 - $p_{ii}$ is by definition 0, no self similarity
- probability that point i "picks" point j as its neighbor based on a **Gaussian distribution centered at i**:

$$p_{j|i} = \frac{\exp(-\beta_i D_{ij})}{\sum_{k \ne i} \exp(-\beta_i D_{ik})}$$

---

### 4. **Calculating Optimal Variance For Each Point**

``` python
 sigma(perplexity, dist_m):....
 ```

- Finds the optimal variance ($\sigma$) for each data sample to match the desired **perplexity ($Perp$)**. We
use **binary search** to find $\sigma$ such that the **Shannon entropy ($H$)**  of the conditional distribution matches.
- for simplicity of calculation the binary search is performed on $\beta$ rather than $\sigma$ ($\beta=\frac{1}{2 \sigma^2}$)
- $\beta_{max}$ is initially set to 10 and $\beta_{min}$ to 0.01, so that the entropy is not very close to zero

Optimal $\sigma$ is calculated through **BINARY SEARCH**:
- **Initialize search range** for $\sigma_i$ or $\beta_i$​:  
    e.g. $\beta_{max}$ = $10$, $\beta_{min}$= $0.01$
- **Repeat**:
    - Set $σ_i$=$\frac{σ_{min}+σ_{max}}{2}$ or $\beta_i$=$\frac{\beta_{min}+\beta_{max}}{2}$
    - Compute the probabilities $p_{j|i}$​ using current $\sigma_i$​
    - Compute entropy $H(P_i)$
    - If $H(P_i)>log⁡_2(Perp)$:
        → Entropy too high → probabilities too flat → **decrease** $\sigma_i$ or **increase** $\beta$
        → Set $\sigma_{\text{max}}$ = $\sigma_i$ or $\beta_{min}=\beta_i$
    - Else:  
        → Entropy too low → probabilities too sharp → **increase** $\sigma_i$ or **decrease** $\beta$
        → Set $\sigma_{\text{min}} = \sigma_i$ or $\beta_{max}=\beta_i$​
- **Stop when** $|H(P_i) - \log_2(\text{Perp})| < \epsilon$ (tolerance)


$$Perp (P_i) = 2^{H(P_i)}$$

$$
H(P_i)=-\sum_j P_{j|i}log_2{ p_{j|i}}
$$

$$\beta_i=\frac{\beta_{min}+\beta_{max}}{2}$$

---
### 5. **Making The Assymetruc Matrix Symmetric**

```python
symmetrize_matrix(m):...
```

- makes a matrix symmetrical by averaging it with its transpose. In t-SNE, it ensures that the  strength between any two points is mutual and consistent.
- used to symmetrize the **asymmetric probability distribution** of high dimension
 - **Large $p_{ij}$**: neighbors in high-D
- **Small $p_{ij}$**: far apart

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

---

### 6. **Initializing Low Dimension Embeddings**

```python
 init_embed(n_components):....
 ```

- Randomly initializing low dimensional embeddings(y) with mean 0 and some noise to avoid same gradients
- **n_components**: specifes the number of dimensions in low dimension embeddings
> for eg: `n_components`=2 for 2D representation

$$ \{ y_1,y_2,y_3.....y_n \} $$
 
 ---
### 7. **Calculating The Low Dimension Matrix**

```python
 low_dim_prob_matrix():...
 ```

- Computes matrix Q with pairwise probabilities in the low-dimensional embedding using **student t-distribution**
- This gives a **symmetric matrix** and need not to symmetrize again like in high dimension probability matrix
- $q_{i|j}$ is globally normalized
- $q_{ii}$ is by definition 0 as we are not looking for self similarity
- **Large $q_{ij}$**: they’re currently neighbors
- **Small $q_{ij}$​**: they’re currently far apart

$$q_{ij} = \frac{(1 + D_{ij})^{-1}}{\sum_{k \ne l} (1 + D_{kl})^{-1}}$$

---

### 8. **Defining Gradient Descent**

``` python
gradient_descent(y,p,q):...
```

- Computes the gradient of **KL divergence** between P and Q with respect to low-dimensional positions y.
- It is the core of optimization—updates y to minimize divergence.

**SYMBOLIC MEANING**
 - **If $p_{ij} \gg q_{ij}$** → model is underestimating the similarity → **pull** points closer
- **If $p_{ij} \ll q_{ij}$​** → model is overestimating → **push** them apart
- $(y_i−y_j)$: A vector (force vector) pointing **away** from $y_j$
    - Positive coefficient → **pull $y_i$​ toward $y_j$​**
    - Negative coefficient → **push $y_i$​ away from $y_j$​**
- $(1 + \|y_i - y_j\|^2)^{-1}$: This **scales** the force based on distance
	- Farther points → smaller force
	- Close points → stronger force
- acts like spring
  
$$
\frac{\partial \text{KL}}{\partial y_i} = 4 \sum_{j \ne i} (p_{ij} - q_{ij}) \cdot \frac{y_i - y_j}{1 + \|y_i - y_j\|^2}
$$

---

### 9. **Defining Momentum With Gradient Descent**

``` python
update_with_momentum(learning_rate,momentum,y,P,Q,gains):...
```

- Updates positions of y by applying gradient descent with **momentum** and **gains**.
- It adjusts gains based on sign changes in gradient direction.
  
$$
\begin{aligned}
v_i^{(t+1)} &= \mu \cdot v_i^{(t)} - \eta \cdot gain\cdot \frac{\partial \text{KL}}{\partial y_i^{(t)}} \\\\
y_i^{(t+1)} &= y_i^{(t)} + v_i^{(t+1)}
\end{aligned}
$$


>[!NOTE]
> value of **momentum** changes from 0.5 for starting 250 epochs to 0.8 for the rest

---

### 10. **Defining The t-sne Function**

```python
tsne(X,perplexity,learning_rate,early_exaggeration,max_iter):...
```

- Main function that runs the entire **t-SNE** algorithm.


>[!NOTE]
>NumPy Vectorization properties were used to speed up the model

---


## STEPS:

1. Computes high-dimensional distance matrix.
2. Finds optimal sigma for each point.
3. Computes high-dimensional joint probability matrix P.
4. Initializes low-dimensional positions y randomly.
5. Computes low-dimensional joint probability matrix Q.
6. Iteratively updates y using gradient descent with momentum and gains.
7. Applies early exaggeration to improve cluster separation in early iterations.

---
## Comparison

### **TSNE From Scratch**

<img width="563" height="413" alt="image" src="https://github.com/user-attachments/assets/68cd68a8-40a2-440a-866f-06671806a3eb" />

```
CPU times: user 11min 32s, sys: 8min 43s, total: 20min 15s
Wall time: 17min 50s
```


### **sklearn TSNE**

<img width="568" height="413" alt="image" src="https://github.com/user-attachments/assets/60457a37-a5b4-458e-8fe2-9d551e9803e6" />

```
CPU times: user 10.3 s, sys: 32 ms, total: 10.3 s
Wall time: 10.3 s
```

---
## References:

1. https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
2. https://medium.com/@sachinsoni600517/mastering-t-sne-t-distributed-stochastic-neighbor-embedding-0e365ee898ea

