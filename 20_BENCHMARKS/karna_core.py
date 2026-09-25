"""AJO-1: new cell + weights + real-time learn/adapt. NumPy only, no backprop."""
import numpy as np


def kwta(v, k):
    y = np.zeros_like(v)
    idx = np.argpartition(v, -k)[-k:]
    y[idx] = v[idx]
    return (y > 0).astype(np.float64)


class SPLRCell:
    def __init__(self, n=256, k=8, in_dim=32, seed=0):
        rng = np.random.default_rng(seed)
        self.n, self.k = n, k
        self.D = rng.normal(0, np.sqrt(1.0 / in_dim), (n, in_dim)) * (rng.random((n, in_dim)) < 0.5)
        self.Wp = np.zeros((n, n))
        self.u = np.zeros(n)
        self.y = np.zeros(n)
        self.yp = np.zeros(n)
        self.rho = np.full(n, k / n)
        self.h = np.zeros(n)

    def batch_step(self, X):
        Z = X @ self.D.T
        H = np.zeros_like(Z)
        idx = np.argpartition(Z, -self.k, axis=1)[:, -self.k:]
        H[np.arange(X.shape[0])[:, None], idx] = (Z[np.arange(X.shape[0])[:, None], idx] > 0).astype(np.float64)
        return H, Z

    def step(self, x, eta=0.0, consolidate_every=0):
        z = self.D @ x
        p = self.Wp @ self.yp if eta > 0 else 0.0
        yn = kwta(z + self.h, self.k)
        err = z - p if eta > 0 else z * 0.0
        if eta > 0:
            dp = yn - np.tanh(p)
            self.Wp += eta * 0.5 * np.outer(dp, self.yp)
            self.rho = 0.999 * self.rho + 0.001 * yn
            self.h += 0.01 * ((self.k / self.n) - self.rho)
        self.yp, self.y = self.y, yn
        self.u = z
        if consolidate_every > 0 and hasattr(self, "_t"):
            pass
        return yn, err, z

    def consolidate(self, strength=0.02):
        self.D /= (np.linalg.norm(self.D, axis=1, keepdims=True) + 1e-8)
        self.Wp = np.clip(self.Wp, -0.5, 0.5)
        self.h *= (1.0 - strength)


class PSM:
    def __init__(self, m, n, dens=0.05):
        self.m, self.n = m, n
        self.Ws = np.zeros((m, n))
        self.Wf = np.zeros((m, n))
        self.F = np.zeros((m, n))
        self.mu = np.zeros(n)
        self.sd = np.ones(n)
        self.mem_h, self.mem_y = [], []
        self.heads = {}
        self.head_F = {}

    def act(self, h, dense=None, use_episodic=True, task=None):
        feat = dense if dense is not None else (h - self.mu) / (self.sd + 1e-8)
        W = self.heads[task] if task in self.heads else self.Ws
        y = W @ feat + self.Wf @ feat
        if use_episodic and self.mem_h:
            M = np.stack(self.mem_h)
            hn = np.linalg.norm(h) + 1e-8
            s = M @ h / (hn * (np.linalg.norm(M, axis=1) + 1e-8))
            bi = int(np.argmax(s))
            if float(s[bi]) > 0.95:
                return np.clip(np.asarray(self.mem_y[bi], dtype=np.float64), -5.0, 5.0)
        return np.clip(y, -5.0, 5.0)

    def learn(self, h, e, target, dense=None, eta_s=0.02, lmbda=2.0, eta_f=0.0, decay_f=0.995, task=None, margin=0.2):
        feat = dense if dense is not None else (h - self.mu) / (self.sd + 1e-8)
        W = self.heads.setdefault(task, self.Ws.copy()) if task is not None else self.Ws
        F = self.head_F.setdefault(task, np.zeros_like(self.F)) if task is not None else self.F
        lin = W @ feat + self.Wf @ feat
        order = np.argsort(lin)
        pred_label, second = int(order[-1]), int(order[-2])
        true_label = int(np.argmax(target))
        gap = float(lin[true_label] - lin[pred_label]) if pred_label != true_label else float(lin[pred_label] - lin[second])
        self.mu = 0.999 * self.mu + 0.001 * h
        M = 0.0
        if pred_label != true_label or gap < margin:
            self.sd = np.sqrt(0.99 * self.sd ** 2 + 0.01 * (h - self.mu) ** 2) + 1e-8
            P = 1.0 / (1.0 + lmbda * F[true_label])
            upd_t = eta_s * P * np.abs(feat)
            P2 = 1.0 / (1.0 + lmbda * F[pred_label])
            upd_p = eta_s * P2 * np.abs(feat)
            W[true_label] += upd_t * np.sign(feat + 1e-12)
            W[pred_label] -= upd_p * np.sign(feat + 1e-12)
            W /= np.maximum(1, np.linalg.norm(W, axis=1, keepdims=True) / 2.0)
            F[true_label] += (np.abs(feat) > 0.05) * 0.02
            F[pred_label] += (np.abs(feat) > 0.05) * 0.02
            np.clip(F, 0, 1e4, out=F)
            M = 1.0
        self.mem_h.append(h.copy())
        self.mem_y.append(target.copy())
        if len(self.mem_h) > 2000:
            self.mem_h.pop(0)
            self.mem_y.pop(0)
        self._rehearse(n=10, eta_s=eta_s * 0.5, lmbda=lmbda)
        return M

    def _rehearse(self, n=10, eta_s=0.01, lmbda=2.0):
        if len(self.mem_h) < 20:
            return
        labs = np.array([int(np.argmax(t)) for t in self.mem_y])
        uniq = np.unique(labs)
        per = max(1, n // max(len(uniq), 1))
        picks = []
        for u in uniq:
            ii = np.where(labs == u)[0]
            picks.extend(list(np.random.choice(ii, min(per, len(ii)), replace=False)))
        for i in picks[:n]:
            ho = self.mem_h[i]
            to = self.mem_y[i]
            zo = (ho - self.mu) / (self.sd + 1e-8)
            plo = int(np.argmax(self.Ws @ zo + self.Wf @ zo))
            tlo = int(np.argmax(to))
            if plo != tlo:
                Po = 1.0 / (1.0 + lmbda * self.F[tlo])
                self.Ws[tlo] += eta_s * Po * np.abs(zo) * np.sign(zo + 1e-12)
                Pp = 1.0 / (1.0 + lmbda * self.F[plo])
                self.Ws[plo] -= eta_s * Pp * np.abs(zo) * np.sign(zo + 1e-12)
        self.Ws /= np.maximum(1, np.linalg.norm(self.Ws, axis=1, keepdims=True) / 2.0)

    def batch_acts(self, H, dense_Z):
        n = H.shape[0]
        out = np.empty((n, self.m))
        for i in range(n):
            out[i] = self.act(H[i], dense=dense_Z[i])
        return out

    def save(self, path):
        from karna_psm import PSMFile
        PSMFile(self.Ws, self.Wf, self.F, clock=0.0).save(path)
        np.savez_compressed(path.replace(".npz", ".norm.npz"), mu=self.mu, sd=self.sd)

    def load(self, path):
        from karna_psm import PSMFile
        f = PSMFile.load(path)
        self.Ws, self.Wf, self.F = f.Ws, f.Wf, f.F
        try:
            d = np.load(path.replace(".npz", ".norm.npz"), allow_pickle=False)
            self.mu, self.sd = d["mu"], d["sd"]
        except Exception:
            pass


class AJO1:
    def __init__(self, in_dim=32, n=256, k=8, out_dim=8, seed=0):
        self.cell = SPLRCell(n, k, in_dim, seed=seed)
        self.psm = PSM(out_dim, n)
        self.t = 0

    def step(self, x, target=None, task=None):
        self.t += 1
        h, err, z = self.cell.step(x)
        y = self.psm.act(h, dense=z, task=task)
        M = 0.0
        if target is not None:
            e = target - y
            M = self.psm.learn(h, e, target, dense=z, task=task)
            y = self.psm.act(h, dense=z, task=task)
        return {"y": y, "err": float(np.mean(np.abs(err))), "M": M, "active": int(h.sum())}

    def fit(self, X, Y, tasks=None, epochs=2, eta_s=0.02):
        X = np.asarray(X, dtype=np.float64)
        H, Z = self.cell.batch_step(X)
        n = X.shape[0]
        for ep in range(epochs):
            order = np.random.permutation(n)
            for i in order:
                t = np.asarray(Y[i], dtype=np.float64)
                tk = None if tasks is None else tasks[i]
                e = t - self.psm.act(H[i], dense=Z[i], task=tk)
                self.psm.learn(H[i], e, t, dense=Z[i], task=tk, eta_s=eta_s / (1 + ep))
        return {"epochs": epochs, "n": n}

    def save(self, path):
        self.psm.save(path)

    def load(self, path):
        self.psm.load(path)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ajo = AJO1()
    errs = []
    for t in range(300):
        x = rng.normal(0, 1, 32)
        target = np.zeros(8)
        target[t % 8] = 1.0
        r = ajo.step(x, target if t % 3 == 0 else None)
        errs.append(r["err"])
    r2 = ajo.step(rng.normal(0, 1, 32))
    print(f"AJO1_SMOKE_OK steps=300 mean_err={np.mean(errs):.4f} active={r2['active']} M={r2['M']:.3f}")
