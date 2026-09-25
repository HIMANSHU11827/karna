"""AJO text specialist: T1 orthographic -> T2 syntax / T3 semantics -> T6 code / T7 math."""
import hashlib
import re
import numpy as np


def _hash_idx(tok, dim):
    d = hashlib.md5(tok.encode("utf-8")).digest()
    return int.from_bytes(d[:4], "little") % dim


class TextSpecialist:
    def __init__(self, dim=64, seed=0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.sem = rng.normal(0, 1.0 / dim, (dim, dim))
        self.code_keys = {"def", "class", "import", "for", "while", "if", "else",
                          "return", "function", "var", "let", "const", "{", "}", "(", ")"}
        self.math_ops = set("+-*/=<>^%")

    def t1(self, text, n=3):
        s = str(text).lower()
        grams = [s[i:i + n] for i in range(max(len(s) - n + 1, 1))] or [s]
        vec = np.zeros(self.dim)
        for g in grams:
            vec[_hash_idx(g, self.dim)] += 1.0
        return vec / (np.linalg.norm(vec) + 1e-8)

    def t2(self, text):
        toks = str(text).split()
        return {"ntok": len(toks), "depth": max((t.count("(") + t.count("{") for t in toks), default=0),
                "complexity": float(len(set(toks)) / (len(toks) + 1e-8))}

    def t3(self, orth):
        e = np.tanh(self.sem @ orth)
        return e / (np.linalg.norm(e) + 1e-8)

    def t6(self, text):
        s = str(text)
        lines = s.splitlines() or [s]
        indent = sum(len(l) - len(l.lstrip()) for l in lines) / max(len(lines), 1)
        toks = re.findall(r"[A-Za-z_]+|[{}()]", s)
        hits = sum(1 for t in toks if t in self.code_keys)
        return {"code_score": float(hits / (len(toks) + 1e-8)), "indent": float(indent)}

    def t7(self, text):
        s = str(text)
        digs = [float(m) for m in re.findall(r"-?\d+\.?\d*", s)]
        ops = sum(1 for c in s if c in self.math_ops)
        mag = float(np.log10(max(max([abs(d) for d in digs], default=0.0), 1.0)))
        return {"numbers": digs[:8], "n_ops": ops, "magnitude": mag}

    def encode(self, text):
        orth = self.t1(text)
        syn = self.t2(text)
        sem = self.t3(orth)
        code = self.t6(text)
        math = self.t7(text)
        is_code = code["code_score"] > 0.15 or code["indent"] > 1.5
        is_math = len(math["numbers"]) > 0 and math["n_ops"] > 0
        sal = float(min(1.0, np.linalg.norm(orth) + 0.1 * syn["complexity"]))
        return {"embedding": sem.astype(np.float32), "syntax": syn, "semantics": sem.astype(np.float32),
                "code_score": code, "math_value": math, "is_code": is_code, "is_math": is_math,
                "salience": sal}


if __name__ == "__main__":
    sp = TextSpecialist()
    for s in ["hello world brain learns fast", "def foo(x):\n    return x * 2 + 1", "solve 24 * 17 + 5 = ?"]:
        r = sp.encode(s)
        print(f"text={s[:24]!r} code={r['is_code']} math={r['is_math']} sal={r['salience']:.2f} emb={r['embedding'].shape}")
