"""AJO AudioSpecialist: Core/Belt/Parabelt/TVA auditory hierarchy (NumPy only)."""
import numpy as np


class AudioSpecialist:
    """Hierarchical audio encoder.

    Core: STRF-like spectral filters on STFT magnitudes.
    Belt: timbre/modulation pooling (nonlinear + temporal stats).
    Parabelt: phoneme posteriors via linear projection + softmax.
    TVA: speaker embedding via linear projection + L2 norm.
    """

    def __init__(self, sr=16000, n_core=32, n_belt=24, n_phon=39,
                 spk_dim=16, frame=256, hop=128, seed=0):
        self.sr, self.frame, self.hop = sr, frame, hop
        rng = np.random.default_rng(seed)
        n_freq = frame // 2 + 1
        self.core_filters = rng.standard_normal((n_core, n_freq)) / np.sqrt(n_freq)
        self.belt_mat = rng.standard_normal((n_belt, n_core)) / np.sqrt(n_core)
        self.parabelt_mat = rng.standard_normal((n_phon, n_belt)) / np.sqrt(n_belt)
        self.tva_mat = rng.standard_normal((spk_dim, n_belt)) / np.sqrt(n_belt)

    @staticmethod
    def _softmax(x):
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    @staticmethod
    def _norm(v):
        return v / (np.linalg.norm(v) + 1e-8)

    def _core(self, wave):
        n = max(0, (len(wave) - self.frame) // self.hop + 1)
        idx = (np.arange(self.frame)[None, :] + self.hop * np.arange(n)[:, None])
        frames = wave[idx] * np.hanning(self.frame + 1)[:self.frame]
        mag = np.abs(np.fft.rfft(frames, axis=1))
        act = np.log1p(np.maximum(mag @ self.core_filters.T, 0.0))
        return act  # (T, n_core)

    def _belt(self, core_act):
        hid = np.tanh(core_act @ self.belt_mat.T)
        pooled = np.concatenate([hid.mean(0), hid.std(0) + 1e-6])
        w = pooled.size
        proj = np.resize(self.belt_mat.mean(0), w) if False else pooled[:self.belt_mat.shape[0]]
        return hid, self._norm(proj)

    def _parabelt(self, belt_emb):
        return self._softmax(self.parabelt_mat @ belt_emb)

    def _tva(self, belt_emb):
        return self._norm(self.tva_mat @ belt_emb)

    def encode(self, wave):
        """Encode 1D waveform -> dict{embedding, phonemes, speaker, salience}."""
        wave = np.asarray(wave, dtype=np.float64).ravel()
        core_act = self._core(wave)
        _, belt_emb = self._belt(core_act)
        phonemes = self._parabelt(belt_emb)
        speaker = self._tva(belt_emb)
        sal = np.linalg.norm(core_act, axis=1)
        salience = (sal - sal.min()) / (np.ptp(sal) + 1e-8)
        return {"embedding": belt_emb, "phonemes": phonemes,
                "speaker": speaker, "salience": salience}


if __name__ == "__main__":
    spec = AudioSpecialist()
    wave = np.random.default_rng(0).standard_normal(8000)
    out = spec.encode(wave)
    for k, v in out.items():
        print(k, np.shape(v))
