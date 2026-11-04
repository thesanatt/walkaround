// Python source lines coloured the way VS Code's Dark+ theme colours them, for the math panel. The classes are the
// ones the video's math beat used (v6/product/extract_math.py): keywords and control words by the Python grammar,
// names by a light version of Pylance's semantic colours (a call is a function, a capitalised name or a builtin type
// is a type, an upper-case name is a constant), brackets by nesting depth.

export type TokenClass = 'kw' | 'ctrl' | 'fn' | 'var' | 'const' | 'type' | 'num' | 'str' | 'com' | 'op' | 'b0' | 'b1' | 'b2';
export type Token = [string, TokenClass];

const KW = new Set(['def', 'class', 'lambda', 'None', 'True', 'False', 'and', 'or', 'not', 'in', 'is']);
const CTRL = new Set(['if', 'elif', 'else', 'for', 'while', 'return', 'try', 'except', 'finally', 'with', 'break', 'continue', 'pass', 'raise', 'import', 'from', 'as', 'yield', 'await', 'async', 'assert', 'del']);
const TYPES = new Set(['float', 'int', 'str', 'bool', 'list', 'dict', 'set', 'tuple', 'np', 'cv2', 'math', 'geometry', 'rules']);

const PATTERN = /(#.*$)|([rbfu]*"(?:[^"\\]|\\.)*"|[rbfu]*'(?:[^'\\]|\\.)*')|(\d+(?:\.\d+)?(?:e[-+]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|([([{])|([)\]}])|(\s+)|(.)/g;

export function tokenize(line: string): Token[] {
  const out: Token[] = [];
  let depth = 0;
  const push = (text: string, k: TokenClass) => {
    const last = out[out.length - 1];
    if (last && last[1] === k && k !== 'b0' && k !== 'b1' && k !== 'b2') last[0] += text;
    else out.push([text, k]);
  };
  for (const m of line.matchAll(PATTERN)) {
    const [text, com, str, num, name, open, close, space] = m;
    if (com) push(text, 'com');
    else if (str) push(text, 'str');
    else if (num) push(text, 'num');
    else if (name) {
      const rest = line.slice((m.index ?? 0) + text.length);
      const called = /^\s*\(/.test(rest);
      const k: TokenClass = KW.has(name)
        ? 'kw'
        : CTRL.has(name)
          ? 'ctrl'
          : TYPES.has(name) || /^[A-Z][a-z]/.test(name)
            ? 'type'
            : name.length > 1 && name === name.toUpperCase() && /[A-Z]/.test(name)
              ? 'const'
              : called
                ? 'fn'
                : 'var';
      push(text, k);
    } else if (open) {
      out.push([text, `b${depth % 3}` as TokenClass]);
      depth += 1;
    } else if (close) {
      depth = Math.max(0, depth - 1);
      out.push([text, `b${depth % 3}` as TokenClass]);
    } else if (space) push(text, 'op');
    else push(text, 'op');
  }
  return out;
}
