// main.ts
//
// Entry point for the modularized microgpt. Wires together:
//   io -> tokenizer -> gpt -> optimizer -> train -> sample
//
// Run with:
//   npx tsx src/microgpt/main.ts
//
// Put a names.txt/input.txt file in the working directory if you don't
// want to download the dataset yourself.

import { rng } from "./rng.js";
import { loadDocs } from "./io.js";
import { buildTokenizer } from "./tokenizer.js";
import { makeConfig } from "./config.js";
import { makeGPT } from "./gpt.js";
import { Adam } from "./optimizer.js";
import { train } from "./train.js";
import { sampleMany } from "./sample.js";

const docs = loadDocs("input.txt");

rng.shuffle(docs);

console.log(`num docs: ${docs.length}`);

const tok = buildTokenizer(docs);

console.log(`vocab size: ${tok.vocabSize}`);

const cfg = makeConfig(tok.vocabSize);

const gpt = makeGPT(cfg);

console.log(`num params: ${gpt.params.length}`);

const opt = new Adam(gpt.params, {
  learningRate: cfg.learningRate,
  beta1: cfg.beta1,
  beta2: cfg.beta2,
  epsAdam: cfg.epsAdam,
});

train(gpt, opt, tok, docs);

sampleMany(gpt, tok);
