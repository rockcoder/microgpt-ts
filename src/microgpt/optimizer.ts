// optimizer.ts
//
// Adam with linear LR decay. Owns its moment state; call step() once per
// training step to update the params in place (and zero their grads).

import { Value } from "./value.js";

export type AdamConfig = {
  learningRate: number;
  beta1: number;
  beta2: number;
  epsAdam: number;
};

export class Adam {
  private m: number[];
  private v: number[];

  constructor(
    private params: Value[],
    private cfg: AdamConfig
  ) {
    this.m = new Array<number>(params.length).fill(0);
    this.v = new Array<number>(params.length).fill(0);
  }

  step(step: number, numSteps: number): void {
    const { learningRate, beta1, beta2, epsAdam } = this.cfg;

    const lrT =
      learningRate *
      (1 - step / numSteps);

    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];

      this.m[i] =
        beta1 * this.m[i] +
        (1 - beta1) * p.grad;

      this.v[i] =
        beta2 * this.v[i] +
        (1 - beta2) * p.grad ** 2;

      const mHat =
        this.m[i] /
        (1 - beta1 ** (step + 1));

      const vHat =
        this.v[i] /
        (1 - beta2 ** (step + 1));

      p.data -=
        lrT *
        mHat /
        (Math.sqrt(vHat) + epsAdam);

      // Reset gradient for next iteration.
      p.grad = 0;
    }
  }
}
