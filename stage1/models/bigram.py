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

counts = counts + 1; # add one smoothing (to  prevent divsion by 0)

probs = counts / counts.sum(axis=1, keepdims=True)


# deteministic generation ( highest proababilty is considered )
char = 's'
c = tokenizer.stoi[char];

next_id = np.argmax(probs[c])
next_char = tokenizer.itos[next_id]

print("next_char:", repr(next_char))

# autoregressive generation (using previously generated tokens to determine next )

def generate(start_char, num_chars):
    current_id = tokenizer.stoi[start_char]
    result = [start_char]

    for _ in range(num_chars):
        probabilities = probs[current_id]

        next_id = np.random.choice(
            vocab_size,
            p = probabilities
        )

        result.append(tokenizer.itos[next_id])

        current_id = next_id;

    return "".join(result)

print("Next tokens using autoregressive gen: \n" + generate("h", 100))
