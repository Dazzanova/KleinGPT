import numpy as np

from stage1.tokenizer.char_tokenizer import Tokenizer

with open("stage1/data/input.txt", "r") as f:
    text = f.read()

tokenizer = Tokenizer(text);

vocab = sorted(set(text))
vocab_size = len(tokenizer.vocab)

counts = np.zeros(
    (vocab_size, vocab_size),
     dtype=np.int32
)

for i in range(len(text) - 1):
    current_id = tokenizer.stoi[text[i]]
    next_id    = tokenizer.stoi[text[i+1]]

    counts[current_id][next_id] += 1

counts = counts + 1;

probs = counts / counts.sum(axis=1, keepdims=True)

print(probs)
