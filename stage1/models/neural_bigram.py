import numpy as np

from stage1.tokenizer.char_tokenizer import Tokenizer

with open("stage1/data/input.txt", "r") as f:
    text = f.read()

tokenizer = Tokenizer(text)

vocab_size = len(tokenizer.vocab)

W = np.random.randn(vocab_size, vocab_size)

print(W.shape)

# def softmax(x):
#     exp_x = np.exp(x)
#     return exp_x / np.sum(exp_x)
# above fn is not stable, exponential incraese when large inputs
# hence, we use numerically stable softmax
# trick: subtract max of x from every number

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


ids = [tokenizer.stoi[ch] for ch in text]

x = np.array(ids[:-1])
y = np.array(ids[1:])

# initializing learning rate
lr = 0.1
epochs = 500

current_id = tokenizer.stoi['h']
target_id = tokenizer.stoi['e']

probs = softmax(W[current_id])

print("Before:", probs[target_id])

for epoch in range(epochs):
    total_loss = 0

    for current_id, target_id in zip(x, y):
        logits = W[current_id]

        probs = softmax(logits)
        loss = -np.log(probs[target_id])
        total_loss += loss

        gradient = probs.copy()
        gradient[target_id] = gradient[target_id] - 1;

        W[current_id] = W[current_id] - lr*gradient

    avg_loss = total_loss / len(x)
    if epoch % 10 == 0:
        print("Epoch: ", epoch + 10, "Loss: ", avg_loss)

probs = softmax(W[current_id])

print("After:", probs[target_id])

def generate(start_char, num_chars):
    current_id = tokenizer.stoi[start_char]
    result = [start_char]

    for _ in range(num_chars):

        probabilities = softmax(W[current_id])

        next_id = np.random.choice(
            vocab_size,
            p=probabilities
        )

        result.append(tokenizer.itos[next_id])

        current_id = next_id

    return "".join(result)

print(generate("h", 100))
