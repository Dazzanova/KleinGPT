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


def train_step(context, target, learning_rate):

    global embedding, w1, b1, w2, b2

    vectors = embedding[context]
    flattened = vectors.flatten()

    hidden_pre = flattened @ w1 + b1
    hidden = np.maximum(0, hidden_pre)

    logits = hidden @ w2 + b2
    probs = softmax(logits)

    loss = -np.log(probs[target])

    dlogits = probs.copy()
    dlogits[target] -= 1

    dw2 = np.outer(hidden, dlogits)
    db2 = dlogits

    dhidden = dlogits @ w2.T

    drelu = hidden_pre > 0
    dhidden_pre = dhidden * drelu

    dw1 = np.outer(flattened, dhidden_pre)
    db1 = dhidden_pre

    dflattened = dhidden_pre @ w1.T

    dvectors = dflattened.reshape(
        context_size,
        embedding_dim
    )

    dembedding = np.zeros_like(embedding)

    for i, token_id in enumerate(context):
        dembedding[token_id] += dvectors[i]

    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2

    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1

    embedding -= learning_rate * dembedding

    return loss


def generate(start_text, num_chars):

    context = [
        tokenizer.stoi[ch]
        for ch in start_text
    ]

    result = list(start_text)

    for _ in range(num_chars):

        context_ids = context[-context_size:]

        vectors = embedding[context_ids]
        flattened = vectors.flatten()

        hidden_pre = flattened @ w1 + b1
        hidden = np.maximum(0, hidden_pre)

        logits = hidden @ w2 + b2
        probs = softmax(logits)

        next_id = np.random.choice(
            vocab_size,
            p=probs
        )

        result.append(tokenizer.itos[next_id])

        context.append(next_id)

    return "".join(result)


learning_rate = 0.01
epochs = 100


for epoch in range(epochs):

    total_loss = 0

    for context, target in zip(x, y):

        loss = train_step(
            context,
            target,
            learning_rate
        )

        total_loss += loss

    average_loss = total_loss / len(x)

    if epoch % 10 == 0:
        print(
            "Epoch:",
            epoch,
            "Loss:",
            average_loss
        )


print()
print("Generated text:")
print(generate("hel", 200))
