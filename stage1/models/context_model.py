import numpy as np

from stage1.tokenizer.char_tokenizer import Tokenizer

with open("stage1/data/input.txt", "r") as f:
    text = f.read()

tokenizer = Tokenizer(text)

vocab_size = len(tokenizer.vocab)
context_size = 3
embedding_dim = 8
hidden_dim = 32

embedding = np.random.randn(
    vocab_size,
    embedding_dim
) * 0.01

w1 = np.random.randn(
    context_size * embedding_dim,
    hidden_dim
) * 0.01

b1 = np.zeros(hidden_dim)

w2 = np.random.randn(
    hidden_dim,
    vocab_size
) * 0.01

b2 = np.zeros(vocab_size)

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

ids = [tokenizer.stoi[ch] for ch in text]

x = []
y = []

for i in range(len(ids) - context_size):
    context = ids[i:i + context_size]
    target = ids[i + context_size]

    x.append(context)
    y.append(target)

x = np.array(x)
y = np.array(y)

context = x[0]
target = y[0]

print("Context:", context)
print("Target:", target)

vectors = embedding[context]
print(vectors.shape)

flattened = vectors.flatten()
print(flattened.shape)

hidden_pre = flattened @ w1 + b1
hidden = np.maximum(0, hidden_pre)

logits = hidden @ w2 + b2
probs = softmax(logits)

print("Hidden shape:", hidden.shape)
print("Logits shape:", logits.shape)
print("Probability sum:", probs.sum())

loss = -np.log(probs[target])

print("Target probability:", probs[target])
print("Loss:", loss)

dlogits = probs.copy()
dlogits[target] = dlogits[target] - 1

dw2 = np.outer(hidden, dlogits)
db2 = dlogits

dhidden = dlogits @ w2.T

drelu = (hidden_pre > 0)
dhidden_pre = dhidden * drelu

dw1 = np.outer(flattened, dhidden_pre)
db1 = dhidden_pre

dflattened = dhidden_pre @ w1.T

dvectors = dflattened.reshape(context_size, embedding_dim)

dembedding = np.zeros_like(embedding)

for i, token_id in enumerate(context):
    dembedding[token_id] += dvectors[i]

learning_rate = 0.1

w2 -= learning_rate * dw2
b2 -= learning_rate * db2

w1 -= learning_rate * dw1
b1 -= learning_rate * db1

embedding -= learning_rate * dembedding

vectors = embedding[context]
flattened = vectors.flatten()

hidden_pre = flattened @ w1 + b1
hidden = np.maximum(0, hidden_pre)

logits = hidden @ w2 + b2
probs = softmax(logits)

new_loss = -np.log(probs[target])

print("After update:")
print("Target probability:", probs[target])
print("Loss:", new_loss)
