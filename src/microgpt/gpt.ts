// gpt.ts
//
// Minimal GPT: token + position embeddings, pre-norm transformer blocks with
// multi-head causal attention and a ReLU MLP, and an LM head. Parameters live
// in a flat state dict and a flattened param list for the optimizer.

import { Value } from "./value.js";
import { rng } from "./rng.js";
import {
  Matrix,
  Vector,
  linear,
  rmsnorm,
  softmax,
  sumValues,
} from "./nn.js";
import { Config, KVCache, headDim } from "./config.js";

export type StateDict = Record<string, Matrix>;

export type GPT = {
  cfg: Config;
  stateDict: StateDict;
  params: Value[];
};

function matrix(
  nout: number,
  nin: number,
  std = 0.08
): Matrix {
  return Array.from(
    { length: nout },
    () =>
      Array.from(
        { length: nin },
        () => new Value(rng.gauss(0, std))
      )
  );
}

function flatten(stateDict: StateDict): Value[] {
  const params: Value[] = [];

  for (const mat of Object.values(stateDict)) {
    for (const row of mat) {
      for (const p of row) {
        params.push(p);
      }
    }
  }

  return params;
}

export function makeGPT(cfg: Config): GPT {
  const stateDict: StateDict = {
    wte: matrix(cfg.vocabSize, cfg.nEmbd),
    wpe: matrix(cfg.blockSize, cfg.nEmbd),
    lm_head: matrix(cfg.vocabSize, cfg.nEmbd),
  };

  for (let i = 0; i < cfg.nLayer; i++) {
    stateDict[`layer${i}.attn_wq`] =
      matrix(cfg.nEmbd, cfg.nEmbd);

    stateDict[`layer${i}.attn_wk`] =
      matrix(cfg.nEmbd, cfg.nEmbd);

    stateDict[`layer${i}.attn_wv`] =
      matrix(cfg.nEmbd, cfg.nEmbd);

    stateDict[`layer${i}.attn_wo`] =
      matrix(cfg.nEmbd, cfg.nEmbd);

    stateDict[`layer${i}.mlp_fc1`] =
      matrix(4 * cfg.nEmbd, cfg.nEmbd);

    stateDict[`layer${i}.mlp_fc2`] =
      matrix(cfg.nEmbd, 4 * cfg.nEmbd);
  }

  return { cfg, stateDict, params: flatten(stateDict) };
}

export function forward(
  gpt: GPT,
  tokenId: number,
  posId: number,
  keys: KVCache,
  values: KVCache
): Value[] {
  const { cfg, stateDict } = gpt;
  const hd = headDim(cfg);

  const tokEmb = stateDict.wte[tokenId];
  const posEmb = stateDict.wpe[posId];

  let x: Vector = tokEmb.map((t, i) =>
    t.add(posEmb[i])
  );

  x = rmsnorm(x);

  for (let li = 0; li < cfg.nLayer; li++) {
    // -------------------------------------------------------------------------
    // Multi-head attention
    // -------------------------------------------------------------------------

    const xResidual = x;

    x = rmsnorm(x);

    const q = linear(
      x,
      stateDict[`layer${li}.attn_wq`]
    );

    const k = linear(
      x,
      stateDict[`layer${li}.attn_wk`]
    );

    const v = linear(
      x,
      stateDict[`layer${li}.attn_wv`]
    );

    keys[li].push(k);
    values[li].push(v);

    const xAttn: Vector = [];

    for (let h = 0; h < cfg.nHead; h++) {
      const hs = h * hd;

      const qH = q.slice(hs, hs + hd);

      const kH = keys[li].map((ki) =>
        ki.slice(hs, hs + hd)
      );

      const vH = values[li].map((vi) =>
        vi.slice(hs, hs + hd)
      );

      const attnLogits = kH.map((kt) => {
        const dot = sumValues(
          qH.map((qj, j) => qj.mul(kt[j]))
        );

        return dot.div(Math.sqrt(hd));
      });

      const attnWeights = softmax(attnLogits);

      for (let j = 0; j < hd; j++) {
        const value = sumValues(
          vH.map((vt, t) =>
            attnWeights[t].mul(vt[j])
          )
        );

        xAttn.push(value);
      }
    }

    x = linear(
      xAttn,
      stateDict[`layer${li}.attn_wo`]
    );

    x = x.map((a, i) => a.add(xResidual[i]));

    // -------------------------------------------------------------------------
    // MLP
    // -------------------------------------------------------------------------

    const mlpResidual = x;

    x = rmsnorm(x);

    x = linear(
      x,
      stateDict[`layer${li}.mlp_fc1`]
    );

    x = x.map((xi) => xi.relu());

    x = linear(
      x,
      stateDict[`layer${li}.mlp_fc2`]
    );

    x = x.map((a, i) => a.add(mlpResidual[i]));
  }

  return linear(x, stateDict.lm_head);
}
