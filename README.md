# microgpt-ts

A tiny, dependency-light GPT implementation in TypeScript. It trains a
character-level Transformer on names using scalar autograd, then generates
new names — all in one readable file.

This is a TypeScript port inspired by
[Andrej Karpathy's `microgpt.py`](https://gist.github.com/karpathy/059f1e2c).

## Quick start

```bash
npm install
npm start
```

The included `input.txt` is the training dataset (one name per line). To train
on your own data, replace it with another newline-separated text file.

## Scripts

```bash
npm run dev        # Run the model
npm start          # Run the model
npm run typecheck  # Check TypeScript types
```

The program prints training loss followed by 20 generated samples. Runs use a
seeded random number generator, so results are reproducible for the same
dataset and configuration.

## How it works

`src/microgpt.ts` implements the complete pipeline:

- character tokenizer with a beginning-of-sequence token
- scalar automatic differentiation
- a small Transformer with attention and an MLP
- Adam optimization
- autoregressive text generation

The model configuration and training length are deliberately kept small and
easy to experiment with near the top of the source file.
