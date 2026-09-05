import numpy as np
from stage1.tokenizer.char_tokenizer import Tokenizer


with open("stage1/data/input.txt", "r") as f:
    text = f.read()


tokenizer = Tokenizer(text)

vocab_size = len(tokenizer.vocab)
context_size = 3
embedding_dim = 8
hidden_dim = 32

embedding = np.random.randn(vocab_size, embedding_dim) * 0.01
w1 = np.random.randn(context_size * embedding_dim, hidden_dim) * 0.01
b1 = np.zeros(hidden_dim)

w2 = np.random.randn(hidden_dim, vocab_size) * 0.01
b2 = np.zeros(vocab_size)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


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


split = int(0.9 * len(x))

x_train = x[:split]
y_train = y[:split]

x_val = x[split:]
y_val = y[split:]


def train_step(contexts, targets, learning_rate):
    batch_size = len(contexts)

    vectors = embedding[contexts]

    flattened = vectors.reshape(batch_size, -1)

    hidden_pre = flattened @ w1 + b1
    hidden = np.maximum(0, hidden_pre)

    logits = hidden @ w2 + b2

    probs = softmax(logits)

    correct_probs = probs[np.arange(batch_size), targets]

    loss = np.mean(-np.log(correct_probs))

    dlogits = probs.copy()
    dlogits[np.arange(batch_size), targets] -= 1
    dlogits /= batch_size

    dw2 = hidden.T @ dlogits
    db2 = np.sum(dlogits, axis=0)

    dhidden = dlogits @ w2.T

    drelu = hidden_pre > 0
    dhidden_pre = dhidden * drelu

    dw1 = flattened.T @ dhidden_pre
    db1 = np.sum(dhidden_pre, axis=0)

    dflattened = dhidden_pre @ w1.T

    dvectors = dflattened.reshape(
        batch_size,
        context_size,
        embedding_dim
    )

    dembedding = np.zeros_like(embedding)

    np.add.at(
        dembedding,
        contexts,
        dvectors
    )

    w2[:] -= learning_rate * dw2
    b2[:] -= learning_rate * db2

    w1[:] -= learning_rate * dw1
    b1[:] -= learning_rate * db1

    embedding[:] -= learning_rate * dembedding

    return loss


def evaluate(contexts, targets):
    batch_size = len(contexts)

    vectors = embedding[contexts]

    flattened = vectors.reshape(batch_size, -1)

    hidden_pre = flattened @ w1 + b1
    hidden = np.maximum(0, hidden_pre)

    logits = hidden @ w2 + b2

    probs = softmax(logits)

    correct_probs = probs[np.arange(batch_size), targets]

    loss = -np.log(correct_probs)

    return np.mean(loss)


def generate(start_text, num_chars):
    context = [tokenizer.stoi[ch] for ch in start_text]

    result = list(start_text)

    for _ in range(num_chars):
        context_ids = context[-context_size:]

        vectors = embedding[context_ids]

        flattened = vectors.reshape(1, -1)

        hidden_pre = flattened @ w1 + b1
        hidden = np.maximum(0, hidden_pre)

        logits = hidden @ w2 + b2

        probs = softmax(logits)

        probabilities = probs[0]

        next_id = np.random.choice(
            vocab_size,
            p=probabilities
        )

        result.append(tokenizer.itos[next_id])

        context.append(next_id)

    return "".join(result)


batch_size = 32
learning_rate = 0.1
epochs = 100


for epoch in range(epochs):

    indices = np.random.permutation(len(x_train))

    total_loss = 0
    total_examples = 0

    for start in range(0, len(x_train), batch_size):

        batch_indices = indices[start:start + batch_size]

        contexts = x_train[batch_indices]
        targets = y_train[batch_indices]

        loss = train_step(
            contexts,
            targets,
            learning_rate
        )

        current_batch_size = len(contexts)

        total_loss += loss * current_batch_size
        total_examples += current_batch_size

    train_loss = total_loss / total_examples

    val_loss = evaluate(x_val, y_val)

    if epoch % 10 == 0:
        print(
            "Epoch:",
            epoch,
            "Train Loss:",
            train_loss,
            "Val Loss:",
            val_loss
        )

print()
print("Generated text:")
print(generate("hel", 300))
