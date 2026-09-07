import numpy as np

np.random.seed(42)

seq_len = 4
d_model = 4
num_heads = 2
head_dim = d_model // num_heads


X = np.random.randn(seq_len, d_model)

print("X:")
print(X)

print()
print("X shape:", X.shape)


W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)


Q = X @ W_Q
K = X @ W_K
V = X @ W_V


Q = Q.reshape(seq_len, num_heads, head_dim)
K = K.reshape(seq_len, num_heads, head_dim)
V = V.reshape(seq_len, num_heads, head_dim)


Q = Q.transpose(1, 0, 2)
K = K.transpose(1, 0, 2)
V = V.transpose(1, 0, 2)


print()
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


scores = Q @ K.transpose(0, 2, 1)

scores = scores / np.sqrt(head_dim)


mask = np.triu(
    np.ones((seq_len, seq_len)),
    k=1
)

scores = np.where(
    mask == 1,
    -np.inf,
    scores
)


attention_weights = softmax(scores)


print()
print("Attention weights shape:", attention_weights.shape)

print()
print("Head 1 attention:")
print(attention_weights[0])

print()
print("Head 2 attention:")
print(attention_weights[1])


head_outputs = attention_weights @ V


print()
print("Head outputs shape:", head_outputs.shape)


head_outputs = head_outputs.transpose(1, 0, 2)

output = head_outputs.reshape(seq_len, d_model)


print()
print("Final output shape:", output.shape)

print()
print("Final output:")
print(output)
