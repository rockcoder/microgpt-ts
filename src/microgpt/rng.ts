// rng.ts
//
// Small seeded PRNG (mulberry32) with the sampling helpers the rest of the
// project needs. Kept deterministic so runs are reproducible; note this is
// not bit-for-bit identical to Python's random.Random.

export class RNG {
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

export const rng = new RNG(42);
