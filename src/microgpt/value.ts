// value.ts
//
// Scalar reverse-mode autograd. Each Value tracks its data, its gradient,
// and the (child, local-gradient) pairs from the op that produced it.
// backward() runs a topological sort and accumulates gradients.

export class Value {
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
