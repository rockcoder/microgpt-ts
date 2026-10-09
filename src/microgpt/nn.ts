// nn.ts
//
// Small tensor-of-Values helpers shared by the model: a dense layer, a
// numerically-stable softmax, and RMSNorm.

import { Value } from "./value.js";

export type Vector = Value[];
export type Matrix = Value[][];

export function sumValues(xs: Value[]): Value {
  if (xs.length === 0) {
    return new Value(0);
  }

  return xs.reduce(
    (acc, x) => acc.add(x),
    new Value(0)
  );
}

export function linear(x: Value[], w: Matrix): Value[] {
  return w.map((row) => {
    const terms = row.map((wi, i) =>
      wi.mul(x[i])
    );

    return sumValues(terms);
  });
}

export function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(
    ...logits.map((x) => x.data)
  );

  const exps = logits.map((x) =>
    x.sub(maxVal).exp()
  );

  const total = sumValues(exps);

  return exps.map((e) => e.div(total));
}

export function rmsnorm(x: Value[]): Value[] {
  const ms = sumValues(
    x.map((xi) => xi.mul(xi))
  ).div(x.length);

  const scale = ms
    .add(1e-5)
    .pow(-0.5);

  return x.map((xi) => xi.mul(scale));
}
