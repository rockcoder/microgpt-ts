// tokenizer.ts
//
// Character-level tokenizer. Unique characters in the corpus become tokens;
// BOS (beginning/end of sequence) is an extra token used to seed and stop
// generation.

export type Tokenizer = {
  uchars: string[];
  vocabSize: number;
  BOS: number;
  charToToken: Map<string, number>;
};

export function buildTokenizer(docs: string[]): Tokenizer {
  const uchars = [...new Set(docs.join(""))].sort();
  const BOS = uchars.length;
  const vocabSize = uchars.length + 1;

  const charToToken = new Map<string, number>(
    uchars.map((ch, i) => [ch, i])
  );

  return { uchars, vocabSize, BOS, charToToken };
}

export function encode(
  tok: Tokenizer,
  text: string
): number[] {
  return [...text].map((ch) => {
    const token = tok.charToToken.get(ch);

    if (token === undefined) {
      throw new Error(`Unknown character: ${ch}`);
    }

    return token;
  });
}

export function decode(
  tok: Tokenizer,
  tokens: number[]
): string {
  return tokens.map((t) => tok.uchars[t]).join("");
}
