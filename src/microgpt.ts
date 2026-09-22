// microgpt.ts
//
// TypeScript port of Andrej Karpathy's microgpt.py.
// The structure intentionally stays close to the original:
//   tokenizer -> scalar autograd -> Transformer -> Adam -> inference
//
// Run with:
//   npx tsx microgpt.ts
//
// Put a names.txt/input.txt file next to this script if you don't
// want to download the dataset yourself.

import { readFileSync, existsSync } from "node:fs";

// -----------------------------------------------------------------------------
// Randomness
// -----------------------------------------------------------------------------

// Python's random.Random and JS Math.random() are different PRNGs, so this
// isn't bit-for-bit identical to the Python implementation. We use a small
// seeded PRNG to keep runs reproducible.
class RNG {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  random(): number {
    // mulberry32
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  gauss(mean = 0, std = 1): number {
    // Box-Muller
    let u = 0;
    let v = 0;

    while (u === 0) u = this.random();
    while (v === 0) v = this.random();

    return (
      mean +
      std *
        Math.sqrt(-2 * Math.log(u)) *
        Math.cos(2 * Math.PI * v)
    );
  }

  choiceWeighted(weights: number[]): number {
    const total = weights.reduce((a, b) => a + b, 0);
    let r = this.random() * total;

    for (let i = 0; i < weights.length; i++) {
      r -= weights[i];
      if (r <= 0) return i;
    }

    return weights.length - 1;
  }

  shuffle<T>(xs: T[]): void {
    for (let i = xs.length - 1; i > 0; i--) {
      const j = Math.floor(this.random() * (i + 1));
      [xs[i], xs[j]] = [xs[j], xs[i]];
    }
  }
}

const rng = new RNG(42);

// -----------------------------------------------------------------------------
// Dataset
// -----------------------------------------------------------------------------

let docs: string[];

if (existsSync("input.txt")) {
  docs = readFileSync("input.txt", "utf8")
    .split(/\r?\n/)
    .map((x) => x.trim())
    .filter(Boolean);
} else {
  throw new Error(
    "input.txt not found. Download the names dataset and put it next to microgpt.ts."
  );
}

rng.shuffle(docs);

console.log(`num docs: ${docs.length}`);

// -----------------------------------------------------------------------------
// Tokenizer
// -----------------------------------------------------------------------------

const uchars = [...new Set(docs.join(""))].sort();
const BOS = uchars.length;
const vocabSize = uchars.length + 1;

console.log(`vocab size: ${vocabSize}`);

const charToToken = new Map<string, number>(
  uchars.map((ch, i) => [ch, i])
);

// -----------------------------------------------------------------------------
// Scalar autograd
// -----------------------------------------------------------------------------

class Value {
  data: number;
  grad: number;

  private children: Value[];
  private localGrads: number[];

  constructor(
    data: number,
    children: Value[] = [],
    localGrads: number[] = []
  ) {
    this.data = data;
    this.grad = 0;
    this.children = children;
    this.localGrads = localGrads;
  }

  add(other: Value | number): Value {
    const b = other instanceof Value ? other : new Value(other);

    return new Value(
      this.data + b.data,
      [this, b],
      [1, 1]
    );
  }

  mul(other: Value | number): Value {
    const b = other instanceof Value ? other : new Value(other);

    return new Value(
      this.data * b.data,
      [this, b],
      [b.data, this.data]
    );
  }

  pow(other: number): Value {
    return new Value(
      this.data ** other,
      [this],
      [other * this.data ** (other - 1)]
    );
  }

  log(): Value {
    return new Value(
      Math.log(this.data),
      [this],
      [1 / this.data]
    );
  }

  exp(): Value {
    const e = Math.exp(this.data);

    return new Value(
      e,
      [this],
      [e]
    );
  }

  relu(): Value {
    return new Value(
      Math.max(0, this.data),
      [this],
      [this.data > 0 ? 1 : 0]
    );
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: Value | number): Value {
    return this.add(
      other instanceof Value ? other.neg() : -other
    );
  }

  div(other: Value | number): Value {
    if (other instanceof Value) {
      return this.mul(other.pow(-1));
    }

    return this.mul(1 / other);
  }

  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const buildTopo = (v: Value) => {
      if (visited.has(v)) return;

      visited.add(v);

      for (const child of v.children) {
        buildTopo(child);
      }

      topo.push(v);
    };

    buildTopo(this);

    this.grad = 1;

    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];

      for (let j = 0; j < v.children.length; j++) {
        v.children[j].grad +=
          v.localGrads[j] * v.grad;
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Model configuration
// -----------------------------------------------------------------------------

const nLayer = 1;
const nEmbd = 16;
const blockSize = 16;
const nHead = 4;
const headDim = nEmbd / nHead;

if (!Number.isInteger(headDim)) {
  throw new Error("nEmbd must be divisible by nHead");
}

// -----------------------------------------------------------------------------
// Parameter initialization
// -----------------------------------------------------------------------------

type Matrix = Value[][];
type StateDict = Record<string, Matrix>;

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

const stateDict: StateDict = {
  wte: matrix(vocabSize, nEmbd),
  wpe: matrix(blockSize, nEmbd),
  lm_head: matrix(vocabSize, nEmbd),
};

for (let i = 0; i < nLayer; i++) {
  stateDict[`layer${i}.attn_wq`] =
    matrix(nEmbd, nEmbd);

  stateDict[`layer${i}.attn_wk`] =
    matrix(nEmbd, nEmbd);

  stateDict[`layer${i}.attn_wv`] =
    matrix(nEmbd, nEmbd);

  stateDict[`layer${i}.attn_wo`] =
    matrix(nEmbd, nEmbd);

  stateDict[`layer${i}.mlp_fc1`] =
    matrix(4 * nEmbd, nEmbd);

  stateDict[`layer${i}.mlp_fc2`] =
    matrix(nEmbd, 4 * nEmbd);
}

// Flatten parameters.

const params: Value[] = [];

for (const mat of Object.values(stateDict)) {
  for (const row of mat) {
    for (const p of row) {
      params.push(p);
    }
  }
}

console.log(`num params: ${params.length}`);

// -----------------------------------------------------------------------------
// Math helpers
// -----------------------------------------------------------------------------

function sumValues(xs: Value[]): Value {
  if (xs.length === 0) {
    return new Value(0);
  }

  return xs.reduce(
    (acc, x) => acc.add(x),
    new Value(0)
  );
}

function linear(x: Value[], w: Matrix): Value[] {
  return w.map((row) => {
    const terms = row.map((wi, i) =>
      wi.mul(x[i])
    );

    return sumValues(terms);
  });
}

function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(
    ...logits.map((x) => x.data)
  );

  const exps = logits.map((x) =>
    x.sub(maxVal).exp()
  );

  const total = sumValues(exps);

  return exps.map((e) => e.div(total));
}

function rmsnorm(x: Value[]): Value[] {
  const ms = sumValues(
    x.map((xi) => xi.mul(xi))
  ).div(x.length);

  const scale = ms
    .add(1e-5)
    .pow(-0.5);

  return x.map((xi) => xi.mul(scale));
}

// -----------------------------------------------------------------------------
// GPT
// -----------------------------------------------------------------------------

type Vector = Value[];
type KVCache = Vector[][];

function gpt(
  tokenId: number,
  posId: number,
  keys: KVCache,
  values: KVCache
): Value[] {
  const tokEmb = stateDict.wte[tokenId];
  const posEmb = stateDict.wpe[posId];

  let x = tokEmb.map((t, i) =>
    t.add(posEmb[i])
  );

  x = rmsnorm(x);

  for (let li = 0; li < nLayer; li++) {
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

    // keys[li].push(k as unknown as Value);
    // values[li].push(v as unknown as Value);
    keys[li].push(k);
    values[li].push(v);

    const xAttn: Value[] = [];

    for (let h = 0; h < nHead; h++) {
      const hs = h * headDim;

      const qH = q.slice(
        hs,
        hs + headDim
      );

      const kH = keys[li].map(ki =>
        ki.slice(hs, hs + headDim)
      );

      const vH = values[li].map(vi =>
        vi.slice(hs, hs + headDim)
      );

      const attnLogits = kH.map((kt) => {
        const dot = sumValues(
          qH.map((qj, j) =>
            qj.mul(kt[j])
          )
        );

        return dot.div(Math.sqrt(headDim));
      });

      const attnWeights = softmax(attnLogits);

      for (let j = 0; j < headDim; j++) {
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

    x = x.map((a, i) =>
      a.add(xResidual[i])
    );

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

    x = x.map((a, i) =>
      a.add(mlpResidual[i])
    );
  }

  return linear(
    x,
    stateDict.lm_head
  );
}

// -----------------------------------------------------------------------------
// Adam
// -----------------------------------------------------------------------------

const learningRate = 0.01;
const beta1 = 0.85;
const beta2 = 0.99;
const epsAdam = 1e-8;

const m = new Array<number>(
  params.length
).fill(0);

const v = new Array<number>(
  params.length
).fill(0);

// -----------------------------------------------------------------------------
// Training
// -----------------------------------------------------------------------------

const numSteps = 100; //1000;

for (let step = 0; step < numSteps; step++) {
  const doc = docs[step % docs.length];

  const tokens = [
    BOS,
    ...[...doc].map((ch) => {
      const token = charToToken.get(ch);

      if (token === undefined) {
        throw new Error(`Unknown character: ${ch}`);
      }

      return token;
    }),
    BOS,
  ];

  const n = Math.min(
    blockSize,
    tokens.length - 1
  );

  const keys: KVCache = Array.from(
    { length: nLayer },
    () => []
  );

  const values: KVCache = Array.from(
    { length: nLayer },
    () => []
  );

  const losses: Value[] = [];

  for (let posId = 0; posId < n; posId++) {
    const tokenId = tokens[posId];
    const targetId = tokens[posId + 1];

    const logits = gpt(
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

  // Backprop.
  loss.backward();

  // Adam.
  const lrT =
    learningRate *
    (1 - step / numSteps);

  for (let i = 0; i < params.length; i++) {
    const p = params[i];

    m[i] =
      beta1 * m[i] +
      (1 - beta1) * p.grad;

    v[i] =
      beta2 * v[i] +
      (1 - beta2) * p.grad ** 2;

    const mHat =
      m[i] /
      (1 - beta1 ** (step + 1));

    const vHat =
      v[i] /
      (1 - beta2 ** (step + 1));

    p.data -=
      lrT *
      mHat /
      (Math.sqrt(vHat) + epsAdam);

    // Reset gradient for next iteration.
    p.grad = 0;
  }

  process.stdout.write(
    `step ${(step + 1)
      .toString()
      .padStart(4)} / ${numSteps
      .toString()
      .padStart(4)} | loss ${loss.data.toFixed(4)}\r`
  );
}

// -----------------------------------------------------------------------------
// Inference
// -----------------------------------------------------------------------------

const temperature = 0.5;

console.log(
  "\n--- inference (new, hallucinated names) ---"
);

for (let sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
  const keys: KVCache = Array.from(
    { length: nLayer },
    () => []
  );

  const values: KVCache = Array.from(
    { length: nLayer },
    () => []
  );

  let tokenId = BOS;
  const sample: string[] = [];

  for (
    let posId = 0;
    posId < blockSize;
    posId++
  ) {
    const logits = gpt(
      tokenId,
      posId,
      keys,
      values
    );

    const probs = softmax(
      logits.map((l) =>
        l.div(temperature)
      )
    );

    tokenId = rng.choiceWeighted(
      probs.map((p) => p.data)
    );

    if (tokenId === BOS) {
      break;
    }

    sample.push(uchars[tokenId]);
  }

  console.log(
    `sample ${(sampleIdx + 1)
      .toString()
      .padStart(2)}: ${sample.join("")}`
  );
}