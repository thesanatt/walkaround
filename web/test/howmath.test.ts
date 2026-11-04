import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { COST_LINES, RULE_LINES } from '../src/lib/howmath.ts';

// The How it works page quotes match.py; every quoted line must still be that line of the file.
const source = readFileSync(fileURLToPath(new URL('../../walkaround/match.py', import.meta.url)), 'utf8').split('\n');

describe('the match.py lines on /how', () => {
  for (const excerpt of [COST_LINES, RULE_LINES])
    for (const line of excerpt.lines)
      it(`line ${line.n} is quoted as written`, () => {
        expect(' '.repeat(excerpt.indent) + line.code).toBe(source[line.n - 1]);
      });
});
