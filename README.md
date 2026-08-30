# KleinGPT
GPT-style LLM, built from first principles

KleinGPT is a small language model I'm building from scratch to understand how LLMs work under the hood.

The project starts with simple character-level tokenizer and gradually builds toward a small Transformer, implementing the core ideas manually with Python and NumPy before using frameworks like PyTorch.

## What I've built so far

* Character-level tokenizer
* Token encoding and decoding
* Count-based Bigram language model
* Bigram probability matrix
* Probabilistic text generation
* Neural Bigram model
* Softmax and cross-entropy loss
* Gradient descent
* Manual backpropagation
* Context-based neural language model
* Token embeddings
* Feed-forward neural network
* ReLU activation

Currently working on training the context-based model.

## Roadmap

```text
Tokenizer
   ↓
Bigram Model
   ↓
Neural Bigram
   ↓
Context-based Neural LM
   ↓
Self-Attention
   ↓
Multi-Head Attention
   ↓
Transformer
   ↓
Small GPT-style Model
```

The goal is to understand each step rather than just using an existing LLM implementation.

## Project Structure

```text
KleinGPT/
├── stage1/
│   ├── data/
│   │   └── input.txt
│   ├── tokenizer/
│   │   └── char_tokenizer.py
│   └── models/
│       ├── bigram.py
│       ├── neural_bigram.py
│       └── context_model.py
├── requirements.txt
└── README.md
```

## Setup

Clone the repository:

```bash
git clone https://github.com/Dazzanova/KleinGPT.git
cd KleinGPT
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run the Bigram model:

```bash
python3 -m stage1.models.bigram
```

Run the Neural Bigram:

```bash
python3 -m stage1.models.neural_bigram
```

Run the context model:

```bash
python3 -m stage1.models.context_model
```

## Tech Stack

* Python
* NumPy
Thats it!

## Why I'm building this

I wanted to understand what actually happens inside an LLM instead of treating models like black boxes.

KleinGPT is my attempt to learn it from the bottom up — starting with probability and gradually building toward Transformers.

P.S - Klein means 'small' in German, so its a 'small'GPT!
