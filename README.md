# microgpt-ts

A tiny, dependency-light GPT implementation in TypeScript. It trains a
character-level Transformer on names using scalar autograd, then generates
new names — all in one readable file.

This is a TypeScript port inspired by
[Andrej Karpathy's `microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

## Quick start

```bash
npm install
npm start
```

The included `input.txt` is the training dataset (one name per line). To train
on your own data, replace it with another newline-separated text file.

The default run trains for 1000 steps and takes about 20 seconds. For a quick
smoke run, use `MICROGPT_STEPS=100 npm start`.

## Scripts

```bash
npm run dev        # Run the model
npm start          # Run the model
npm run typecheck  # Check TypeScript types
npm test           # Check the autograd against finite differences
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

`test/gradcheck.ts` verifies the autograd by comparing `Value.backward()`
against central finite differences of the same functions.

The model configuration and training length are deliberately kept small and
easy to experiment with near the top of the source file.
