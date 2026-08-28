class Tokenizer:
    def __init__(self, text):
        self.vocab = sorted(set(text))

        self.stoi = {
            ch: i
            for i, ch in enumerate(self.vocab)
        }

        self.itos = {
            i: ch
            for i, ch in enumerate(self.vocab)
        }

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


with open("stage1/data/input.txt", "r") as f:
    text = f.read()


tokenizer = Tokenizer(text)

print("Vocabulary:", tokenizer.vocab)
print("Vocabulary size:", len(tokenizer.vocab))

encoded = tokenizer.encode("hello")
print("Encoded:", encoded)

decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)
