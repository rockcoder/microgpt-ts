// train.ts
//
// One training step per document: run the model over the token sequence,
// average the per-position cross-entropy losses, backprop, and apply Adam.

import { Value } from "./value.js";
import { softmax, sumValues } from "./nn.js";
import { Config, KVCache, emptyCache } from "./config.js";
import { GPT, forward } from "./gpt.js";
import { Adam } from "./optimizer.js";
import { Tokenizer, encode } from "./tokenizer.js";

export function trainStep(
  gpt: GPT,
  opt: Adam,
  tok: Tokenizer,
  doc: string,
  step: number,
  numSteps: number
): number {
  const { cfg } = gpt;

  const tokens = [
    tok.BOS,
    ...encode(tok, doc),
    tok.BOS,
  ];

  const n = Math.min(
    cfg.blockSize,
    tokens.length - 1
  );

  const keys: KVCache = emptyCache(cfg);
  const values: KVCache = emptyCache(cfg);

  const losses: Value[] = [];

  for (let posId = 0; posId < n; posId++) {
    const tokenId = tokens[posId];
    const targetId = tokens[posId + 1];

    const logits = forward(
      gpt,
      tokenId,
      posId,
      keys,
      values
    );

    const probs = softmax(logits);

    const lossT = probs[targetId].log().neg();

    losses.push(lossT);
  }

  const loss = sumValues(losses).div(n);

  loss.backward();

  opt.step(step, numSteps);

  return loss.data;
}

export function train(
  gpt: GPT,
  opt: Adam,
  tok: Tokenizer,
  docs: string[]
): void {
  const numSteps = gpt.cfg.numSteps;

  for (let step = 0; step < numSteps; step++) {
    const doc = docs[step % docs.length];

    const loss = trainStep(
      gpt,
      opt,
      tok,
      doc,
      step,
      numSteps
    );

    process.stdout.write(
      `step ${(step + 1)
        .toString()
        .padStart(4)} / ${numSteps
        .toString()
        .padStart(4)} | loss ${loss.toFixed(4)}\r`
    );
  }
}
