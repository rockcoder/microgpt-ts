// sample.ts
//
// Autoregressive sampling from a trained model: seed with BOS, sample the
// next token from the temperature-scaled distribution, stop at BOS.

import { rng } from "./rng.js";
import { softmax } from "./nn.js";
import { Config, KVCache, emptyCache } from "./config.js";
import { GPT, forward } from "./gpt.js";
import { Tokenizer, decode } from "./tokenizer.js";

export function sample(
  gpt: GPT,
  tok: Tokenizer,
  temperature?: number
): string {
  const { cfg } = gpt;
  const temp = temperature ?? cfg.temperature;

  const keys: KVCache = emptyCache(cfg);
  const values: KVCache = emptyCache(cfg);

  let tokenId = tok.BOS;
  const sample: number[] = [];

  for (
    let posId = 0;
    posId < cfg.blockSize;
    posId++
  ) {
    const logits = forward(
      gpt,
      tokenId,
      posId,
      keys,
      values
    );

    const probs = softmax(
      logits.map((l) => l.div(temp))
    );

    tokenId = rng.choiceWeighted(
      probs.map((p) => p.data)
    );

    if (tokenId === tok.BOS) break;

    sample.push(tokenId);
  }

  return decode(tok, sample);
}

export function sampleMany(
  gpt: GPT,
  tok: Tokenizer,
  count = 20
): void {
  console.log(
    "\n--- inference (new, hallucinated names) ---"
  );

  for (let i = 0; i < count; i++) {
    console.log(
      `sample ${(i + 1).toString().padStart(2)}: ${sample(gpt, tok)}`
    );
  }
}
