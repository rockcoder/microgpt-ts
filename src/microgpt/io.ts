// io.ts
//
// Filesystem input for the training corpus.

import { readFileSync, existsSync } from "node:fs";

export function loadDocs(path = "input.txt"): string[] {
  if (!existsSync(path)) {
    throw new Error(
      `${path} not found. Download the names dataset and put it next to microgpt.ts.`
    );
  }

  return readFileSync(path, "utf8")
    .split(/\r?\n/)
    .map((x) => x.trim())
    .filter(Boolean);
}
