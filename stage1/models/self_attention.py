import numpy as np


np.random.seed(42)

X = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

print("X:")
print(X)

d_model = X.shape[1]

W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print()
print("Q:")
print(Q)

print()
print("K:")
print(K)

print()
print("V:")
print(V)

scores = Q @ K.T

print()
print("Scores:")
print(scores)

d_k = K.shape[1]

scores = (Q @ K.T) / np.sqrt(d_k)  # scaled dot-product attention

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

attention_weights = softmax(scores)

print()
print("Attention weights:")
print(attention_weights)
