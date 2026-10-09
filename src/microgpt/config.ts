// config.ts
//
// Model and training hyperparameters.

import { Value } from "./value.js";

export type Config = {
  vocabSize: number;
  nLayer: number;
  nEmbd: number;
  nHead: number;
  blockSize: number;
  learningRate: number;
  beta1: number;
  beta2: number;
  epsAdam: number;
  numSteps: number;
  temperature: number;
};

export function makeConfig(
  vocabSize: number
): Config {
  const cfg: Config = {
    vocabSize,
    nLayer: 1,
    nEmbd: 16,
    nHead: 4,
    blockSize: 16,
    learningRate: 0.01,
    beta1: 0.85,
    beta2: 0.99,
    epsAdam: 1e-8,
    numSteps: 100,
    temperature: 0.5,
  };

  if (cfg.nEmbd % cfg.nHead !== 0) {
    throw new Error("nEmbd must be divisible by nHead");
  }

  return cfg;
}

export function headDim(cfg: Config): number {
  return cfg.nEmbd / cfg.nHead;
}

export type KVCache = Value[][][];

export function emptyCache(cfg: Config): KVCache {
  return Array.from({ length: cfg.nLayer }, () => []);
}
