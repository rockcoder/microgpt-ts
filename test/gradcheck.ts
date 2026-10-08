// gradcheck.ts
//
// Finite-difference gradient checks for the scalar autograd in src/microgpt.ts.
// Analytic gradients from Value.backward() are compared against central
// differences of the same function evaluated in plain JS.
//
// Run with: npm test

import {
  Value,
  linear,
  rmsnorm,
  softmax,
  sumValues,
} from "../src/microgpt.ts";

let checked = 0;
let failed = 0;

function close(
  name: string,
  analytic: number,
  numeric: number,
  tol = 1e-6
): void {
  checked++;

  const err = Math.abs(analytic - numeric);
  const ok = err <= tol || err <= tol * Math.abs(numeric);

  if (!ok) failed++;

  console.log(
    `${ok ? "ok  " : "FAIL"} ${name} ` +
      `(analytic=${analytic}, numeric=${numeric})`
  );
}

function numericGrad(
  f: (xs: number[]) => number,
  xs: number[],
  i: number,
  h = 1e-5
): number {
  const up = [...xs];
  const down = [...xs];

  up[i] += h;
  down[i] -= h;

  return (f(up) - f(down)) / (2 * h);
}

// Every op in one smooth composite: add, mul, pow, log, exp, relu, div, sub.
{
  const build = (a: number, b: number) => {
    const A = new Value(a);
    const B = new Value(b);
    const y = A.mul(B)
      .add(A.pow(3))
      .log()
      .exp()
      .relu()
      .div(B.add(2))
      .sub(A.mul(0.5))
      .pow(2);

    return { A, B, y };
  };

  const xs = [1.7, 0.9];
  const f = ([a, b]: number[]) => build(a, b).y.data;
  const g = build(xs[0], xs[1]);

  g.y.backward();

  close("composite d/da", g.A.grad, numericGrad(f, xs, 0));
  close("composite d/db", g.B.grad, numericGrad(f, xs, 1));
}

// A node used more than once must accumulate its gradient, not overwrite it.
{
  const x = new Value(0.6);
  const s = x.add(x.mul(x));
  const y = s.mul(s);

  y.backward();

  const dydx = 2 * (0.6 + 0.6 ** 2) * (1 + 2 * 0.6);

  close("shared subgraph accumulates", x.grad, dydx);
}

// Subgradient convention at the kink: zero on the negative side, zero at zero.
{
  const atZero = new Value(0);
  atZero.relu().backward();
  close("relu(0) grad", atZero.grad, 0);

  const negative = new Value(-3);
  negative.relu().backward();
  close("relu(-3) grad", negative.grad, 0);
}

// softmax: gradient flows through the max-shift-free normalization.
{
  const logits = [1.2, -0.4, 2.1, 0.3];
  const weights = [0.5, -1.5, 2.0, 0.25];

  const f = (ls: number[]) => {
    const max = Math.max(...ls);
    const exps = ls.map((l) => Math.exp(l - max));
    const total = exps.reduce((a, b) => a + b, 0);

    return exps.reduce(
      (acc, e, i) => acc + (e / total) * weights[i],
      0
    );
  };

  const leaves = logits.map((l) => new Value(l));
  const out = sumValues(
    softmax(leaves).map((p, i) => p.mul(weights[i]))
  );

  out.backward();

  leaves.forEach((leaf, i) =>
    close(
      `softmax d/dlogit[${i}]`,
      leaf.grad,
      numericGrad(f, logits, i)
    )
  );
}

// rmsnorm: scale is computed from the whole vector, so every leaf is affected.
{
  const xs = [0.4, -1.1, 0.75, 2.0];
  const weights = [1, 2, -1, 0.5];

  const f = (vs: number[]) => {
    const ms =
      vs.reduce((a, b) => a + b * b, 0) / vs.length;
    const scale = (ms + 1e-5) ** -0.5;

    return vs.reduce(
      (acc, v, i) => acc + v * scale * weights[i],
      0
    );
  };

  const leaves = xs.map((v) => new Value(v));
  const out = sumValues(
    rmsnorm(leaves).map((r, i) => r.mul(weights[i]))
  );

  out.backward();

  leaves.forEach((leaf, i) =>
    close(
      `rmsnorm d/dx[${i}]`,
      leaf.grad,
      numericGrad(f, xs, i)
    )
  );
}

// linear: closed-form gradients for a 2x2 weight matrix and 2-vector input.
{
  const xs = [0.3, -0.8];
  const w = [
    [0.5, -0.2],
    [1.1, 0.9],
  ];

  const leaves = xs.map((v) => new Value(v));
  const mat = w.map((row) => row.map((v) => new Value(v)));

  const out = linear(leaves, mat);
  const y = out[0].mul(2).add(out[1].mul(-3));

  y.backward();

  const expected = [
    [2 * xs[0], 2 * xs[1]],
    [-3 * xs[0], -3 * xs[1]],
  ];

  mat.forEach((row, r) =>
    row.forEach((weight, c) =>
      close(`linear d/dw[${r}][${c}]`, weight.grad, expected[r][c])
    )
  );

  close("linear d/dx[0]", leaves[0].grad, 2 * w[0][0] - 3 * w[1][0]);
  close("linear d/dx[1]", leaves[1].grad, 2 * w[0][1] - 3 * w[1][1]);
}

// sumValues on an empty list is a constant zero and must not blow up.
{
  const empty = sumValues([]);
  empty.backward();
  close("sumValues([]) data", empty.data, 0);
}

console.log(`\n${checked - failed}/${checked} checks passed`);

if (failed > 0) {
  process.exit(1);
}
