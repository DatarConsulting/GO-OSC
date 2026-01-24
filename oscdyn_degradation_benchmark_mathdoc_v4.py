"""
OSCDYN (Datar Consulting) – Benchmark script

This script DOES include:
- A CLI subcommand: `classify` for UEA/UCR-style time-series classification via `aeon`
- A minimal vibration/degradation benchmark interface (IMS/PRONOSTIA/XJTU) kept compact
  (enough to run and extend; uses GO-OSC-like regularization hooks)

CPU-only compatible. GPU optional.
 
Usage (classification):
  pip install aeon scikit-learn torch numpy pandas
  python oscdyn_degradation_benchmark_final.py classify --dataset EthanolConcentration --model go_osc_p_damp --device cpu

Usage (degradation):
  python oscdyn_degradation_benchmark_final.py ims --root /path/IMS --run 0 --out df.csv --out-det det.csv
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from concurrent.futures import ThreadPoolExecutor

# Optional deps for classification / PRONOSTIA
try:
    from aeon.datasets import load_classification  # type: ignore
except Exception:
    load_classification = None

try:
    from sklearn.preprocessing import LabelEncoder  # type: ignore
except Exception:
    LabelEncoder = None

try:
    from scipy.io import loadmat  # type: ignore
except Exception:
    loadmat = None


# ------------------------- utils -------------------------

def seed_all(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    device = str(device)
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ------------------------- basic indicators -------------------------

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    m = float(np.mean(x))
    v = float(np.var(x) + 1e-12)
    z = (x - m) / math.sqrt(v)
    return float(np.mean(z ** 4))


def crest_factor(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.max(np.abs(x)) / (rms(x) + 1e-12))


# ------------------------- IO helpers -------------------------

def _read_numeric_file(path: Union[str, Path], col: int = 0) -> np.ndarray:
    p = Path(path)
    try:
        df = pd.read_csv(p, header=None)
    except Exception:
        df = pd.read_csv(p, header=None, delim_whitespace=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if df.shape[1] == 0:
        raise ValueError(f"No numeric columns in {p}")
    col = min(int(col), df.shape[1] - 1)
    return df.iloc[:, col].to_numpy(dtype=np.float32)


def _read_pronostia_mat(path: Union[str, Path], key: Optional[str] = None, col: int = 0) -> np.ndarray:
    if loadmat is None:
        raise ImportError("scipy required for .mat. Install: pip install scipy")
    d = loadmat(str(path))
    keys = [k for k in d.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No usable keys in {path}")
    if key is None:
        arr = None
        for k in keys:
            v = d[k]
            if isinstance(v, np.ndarray) and v.size >= 128 and np.issubdtype(v.dtype, np.number):
                arr = v
                break
        if arr is None:
            raise ValueError(f"Auto-detect failed in {path}. Keys: {keys}")
    else:
        if key not in d:
            raise KeyError(f"Key '{key}' not found in {path}. Keys: {keys}")
        arr = d[key]
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        c = min(int(col), arr.shape[1] - 1)
        return arr[:, c].astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


def _group_by_parent(files: List[Path], min_files: int) -> List[Tuple[str, List[Path]]]:
    byp: Dict[str, List[Path]] = {}
    for f in files:
        byp.setdefault(str(f.parent), []).append(f)
    runs = [(k, sorted(v)) for k, v in byp.items() if len(v) >= min_files]
    runs.sort(key=lambda kv: len(kv[1]), reverse=True)
    return runs


def discover_runs_ims(root: Union[str, Path], min_files: int = 20) -> List[Tuple[str, List[Path]]]:
    root = Path(root)
    files = list(root.rglob("*.txt")) + list(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No txt/csv under {root}")
    return _group_by_parent(files, min_files=min_files)


def discover_runs_pronostia(root: Union[str, Path], min_files: int = 5) -> List[Tuple[str, List[Path]]]:
    root = Path(root)
    files = list(root.rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No .mat under {root}")
    return _group_by_parent(files, min_files=min_files)


def discover_runs_xjtu(root: Union[str, Path], min_files: int = 5) -> List[Tuple[str, List[Path]]]:
    root = Path(root)
    files = list(root.rglob("*.csv")) + list(root.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No txt/csv under {root}")
    return _group_by_parent(files, min_files=min_files)


# ============================================================
# Classification (UEA/UCR via aeon) + CLI `classify`
# ============================================================

def window_indices(T: int, window_len: int = 256, step: int = 32) -> List[Tuple[int, int]]:
    w = int(window_len)
    st = int(step)
    if w <= 0 or w >= T:
        return [(0, T)]
    st = max(1, min(st, w))
    return [(s, s + w) for s in range(0, T - w + 1, st)] or [(0, min(T, w))]


def extract_raw_windows(X: np.ndarray, idx: List[Tuple[int, int]]) -> np.ndarray:
    """Window extraction.

    IMPORTANT: This is intentionally *non-copying* when possible.
    The old implementation used np.stack([...]) which materializes every window.
    For long sequences, that can explode RAM.
    """

    # Fast path: if idx corresponds to a regular sliding window pattern,
    # use a strided view (zero-copy) and then sub-sample by step.
    if idx:
        w = int(idx[0][1] - idx[0][0])
        if w > 0 and all((b - a) == w for (a, b) in idx):
            starts = np.asarray([a for (a, _) in idx], dtype=np.int64)
            # Detect constant step (or single window)
            if len(starts) <= 1:
                step = 1
            else:
                diffs = np.diff(starts)
                step = int(diffs[0]) if np.all(diffs == diffs[0]) else -1

            if step > 0:
                try:
                    from numpy.lib.stride_tricks import sliding_window_view

                    X = np.asarray(X)
                    if X.ndim != 3:
                        raise ValueError("X must be (N,C,T)")
                    # (N,C,T-w+1,w)
                    win = sliding_window_view(X, window_shape=w, axis=2)
                    # select the same starts we would have gotten from idx
                    win = win[:, :, starts[0] : starts[0] + step * len(idx) : step, :]
                    # -> (N,L,C,W)
                    out = win.transpose(0, 2, 1, 3)
                    # keep float32 without copying if already float32
                    return out if out.dtype == np.float32 else out.astype(np.float32, copy=False)
                except Exception:
                    # Fall through to safe materializing path.
                    pass

    # Safe fallback (materializes windows)
    return np.stack([X[:, :, a:b] for (a, b) in idx], axis=1).astype(np.float32)  # (N,L,C,W)
def _parallel_features_numeric(files: List[Path], col: int, max_workers: Optional[int] = None) -> List[Dict[str, float]]:
    """Parallel IO + feature extraction for txt/csv numeric sensor files."""

    def process_one(f: Path) -> Dict[str, float]:
        x = _read_numeric_file(f, col=col)
        return {"rms": rms(x), "kurtosis": kurtosis(x), "crest": crest_factor(x)}

    # ThreadPool works well here because pandas CSV parsing is mostly IO-bound.
    # If your CSVs are huge and CPU-parsing dominates, you can switch to a ProcessPool.
    max_workers = max_workers or min(32, (os.cpu_count() or 8))
    out: List[Dict[str, float]] = [None] * len(files)  # type: ignore
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_one, f): i for i, f in enumerate(files)}
        for fut, i in futs.items():
            out[i] = fut.result()
    return out


def _parallel_features_pronostia(files: List[Path], key: Optional[str], col: int, max_workers: Optional[int] = None) -> List[Dict[str, float]]:
    """Parallel IO + feature extraction for PRONOSTIA .mat files."""

    def process_one(f: Path) -> Dict[str, float]:
        x = _read_pronostia_mat(f, key=key, col=col)
        return {"rms": rms(x), "kurtosis": kurtosis(x), "crest": crest_factor(x)}

    max_workers = max_workers or min(32, (os.cpu_count() or 8))
    out: List[Dict[str, float]] = [None] * len(files)  # type: ignore
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_one, f): i for i, f in enumerate(files)}
        for fut, i in futs.items():
            out[i] = fut.result()
    return out


def window_lift_mu_std_dmu(Xw: np.ndarray) -> np.ndarray:
    mu = Xw.mean(axis=-1)
    sd = Xw.std(axis=-1) + 1e-6
    dmu = (Xw[..., 1:] - Xw[..., :-1]).mean(axis=-1)
    return np.concatenate([mu, sd, dmu], axis=2).astype(np.float32)  # (N,L,3C)


def load_uea_windows(ds_name: str, window_len: int, step: int, seed: int = 0):
    if load_classification is None:
        raise ImportError("aeon is required. Install: pip install aeon")
    if LabelEncoder is None:
        raise ImportError("scikit-learn is required. Install: pip install scikit-learn")

    Xtr, ytr = load_classification(ds_name, split="train")
    Xte, yte = load_classification(ds_name, split="test")

    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32)

    le = LabelEncoder()
    ytr = le.fit_transform(np.asarray(ytr)).astype(np.int64)
    yte = le.transform(np.asarray(yte)).astype(np.int64)

    N, C, T = Xtr.shape
    idx = window_indices(T, window_len=window_len, step=step)
    Xwtr = extract_raw_windows(Xtr, idx)
    Xwte = extract_raw_windows(Xte, idx)
    Utr = window_lift_mu_std_dmu(Xwtr)
    Ute = window_lift_mu_std_dmu(Xwte)
    return Utr, Ute, Xwtr, Xwte, ytr, yte, int(C)


# ---- Compact LinOSS / GO-OSC classifiers ----


# =============================================================================
# Mathematical core: Geometric / Oscillatory State-Space Blocks
# =============================================================================
#
# This section is the only "math-heavy" part of the script.
# It is written for data scientists who want a practical model, but also want
# a clear mapping to the underlying mathematics.
#
# Notation used below (batch dimension B omitted):
#   - u_t ∈ R^{d_in}         : input at time t
#   - z_t ∈ R^{n_state}      : latent state at time t
#   - A ∈ R^{n×n}            : state transition operator (spectrally controlled)
#   - B ∈ R^{n×d_in}         : input injection
#
# Baseline linear recurrence:
#   z_{t+1} = A z_t + B u_t
#
# To get stable oscillations, we construct A as an orthogonal similarity
# transform of a block-diagonal rotation:
#   A = Pᵀ Q(θ) P
# where:
#   - P is orthogonal (PᵀP = I),
#   - Q(θ) is block-diagonal with 2×2 rotation blocks R(θ_k),
#     giving complex-conjugate eigenvalues e^{± i θ_k}.
#
# Optional stability mechanism:
#   z_{t+1} ← r ⊙ z_{t+1},  with r ∈ (r_min, 1) elementwise
# computed from input context. This encourages contractive behavior without
# destroying oscillations (because Q is orthogonal and r < 1 damps amplitudes).
#
# Optional "invented" upgrade (Implicit diagonal-Jacobian nonlinearity):
#   z_{t+1} - α tanh(z_{t+1}) = r ⊙ (A z_t + B u_t)
# This is solved by a few Newton steps elementwise. The Jacobian with respect
# to z_{t+1} is diagonal:
#   J = 1 - α sech²(z)
# so the Newton update is cheap and stable in practice.
#
# =============================================================================

def skew_symmetric(M: torch.Tensor) -> torch.Tensor:
    """
    Return the skew-symmetric part of a square matrix.

    Math:
        skew(M) = (M - Mᵀ)/2  which satisfies skew(M)ᵀ = -skew(M).

    Why it matters:
        The Cayley transform takes a skew-symmetric matrix Ω and maps it to an
        orthogonal matrix:
            P = (I - εΩ)^{-1} (I + εΩ)
        This is an efficient way to *parameterize* orthogonal matrices with
        unconstrained parameters.
    """
    return 0.5 * (M - M.transpose(-1, -2))


def cayley_transform(Omega: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Cayley transform mapping skew-symmetric Ω to an orthogonal matrix P.

    P(Ω) = (I - εΩ)^{-1} (I + εΩ)

    Properties (for small eps and well-conditioned solve):
      - If Ωᵀ = -Ω then PᵀP = I (P is orthogonal).
      - This parameterization is differentiable and avoids expensive QR at each step.

    Note:
      We solve a linear system rather than forming an explicit inverse.
    """
    n = Omega.shape[-1]
    I = torch.eye(n, device=Omega.device, dtype=Omega.dtype)
    return torch.linalg.solve(I - eps * Omega, I + eps * Omega)


class CayleyOrthogonalParameterization(nn.Module):
    """
    Learnable orthogonal matrix P ∈ O(n) via Cayley transform.

    Parameters:
        n   : dimension of the square matrix
        eps : small scalar controlling the Cayley transform conditioning
        init_scale : scale of the random init

    Forward:
        returns P (n×n) such that approximately PᵀP = I.
    """
    def __init__(self, n: int, eps: float = 1e-4, init_scale: float = 1e-2):
        super().__init__()
        self.n = int(n)
        self.eps = float(eps)
        self.W_unconstrained = nn.Parameter(init_scale * torch.randn(n, n))

    def forward(self) -> torch.Tensor:
        Omega = skew_symmetric(self.W_unconstrained)
        return cayley_transform(Omega, eps=self.eps)


class BlockContextDampingGate(nn.Module):
    """
    Context-dependent elementwise damping r ∈ (r_min, 1).

    Inputs:
        x_blk : (B, nb, C, W) block context (mean/summary of a time window)

    Output:
        r     : (B, n_state) damping factors per latent dimension

    Interpretation:
        The recurrence uses:
            z ← r ⊙ z
        which introduces controlled contraction without requiring A itself to be
        contractive. This is useful when A is (approximately) orthogonal
        (oscillatory) and we still need amplitude control for long sequences.
    """
    def __init__(self, C_in: int, n_state: int, r_min: float = 0.98):
        super().__init__()
        self.r_min = float(r_min)
        self.net = nn.Sequential(
            nn.Conv1d(C_in, 64, 7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, n_state),
            nn.Sigmoid(),  # outputs in (0,1)
        )

    def forward(self, x_blk: torch.Tensor) -> torch.Tensor:
        # x_blk: (B, nb, C, W) → merge blocks by averaging
        x_ctx = x_blk.mean(dim=1)          # (B, C, W)
        s = self.net(x_ctx)                # (B, n_state) in (0,1)
        return self.r_min + (1 - self.r_min) * s


def _rotation_block_diagonal(theta: torch.Tensor, n_state: int) -> torch.Tensor:
    """
    Build a dense block-diagonal rotation matrix Q(θ) ∈ R^{n×n}.

    theta: (n/2,) angles; each produces a 2×2 rotation block.
    Q is orthogonal (QᵀQ = I) and has eigenvalues e^{± i θ_k}.
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    diag_cos = torch.repeat_interleave(cos_t, 2)  # [cos1, cos1, cos2, cos2, ...]
    Q = torch.diag(diag_cos)

    rng = torch.arange(n_state // 2, device=theta.device)
    i = 2 * rng
    j = 2 * rng + 1
    Q[i, j] = -sin_t
    Q[j, i] =  sin_t
    return Q


class OrthogonalOscillatorSSMClassifier(nn.Module):
    """
    Geometric Oscillator State-Space classifier (GO-OSC, math-corrected).

    This fixes a common pitfall: using A = PᵀP which collapses to Identity.
    Instead we use:
        A = Pᵀ Q(θ) P
    where Q(θ) is a bank of 2×2 rotations.

    Forward signature matches the rest of the benchmark:
        forward(u, xw) → logits

    Inputs:
        u  : (B, L, d_in)    windowed raw signal (or features)
        xw : (B, L, C, W)    windowed spectrogram / context tensor (used only for gating)

    Notes for practitioners:
      - With use_damp=True, the gate supplies r ∈ (r_min, 1) to control amplitude.
      - LayerNorm keeps state scale stable across time and batches.
    """
    def __init__(
        self,
        d_in: int,
        C_in: int,
        n_state: int,
        n_classes: int,
        block_len: int = 8,
        eps: float = 1e-4,
        use_damp: bool = False,
        r_min: float = 0.98,
    ):
        super().__init__()
        assert n_state % 2 == 0, "n_state must be even (2×2 oscillator blocks)."
        self.n_state = int(n_state)
        self.block_len = int(block_len)

        # Orthogonal change-of-basis P
        self.P = CayleyOrthogonalParameterization(self.n_state, eps=eps, init_scale=2e-3)

        # Rotation angles (one per 2D block)
        self.theta = nn.Parameter(torch.randn(self.n_state // 2) * 0.1)

        # Input injection Bu_t
        self.B_in = nn.Linear(int(d_in), self.n_state, bias=False)

        self.norm = nn.LayerNorm(self.n_state)
        self.head = nn.Linear(self.n_state, int(n_classes))

        self.damping_gate = BlockContextDampingGate(C_in=C_in, n_state=self.n_state, r_min=r_min) if use_damp else None

    def forward(self, u: torch.Tensor, xw: torch.Tensor) -> torch.Tensor:
        Bsz, L, _ = u.shape
        nb = (L + self.block_len - 1) // self.block_len
        pad = nb * self.block_len - L
        if pad:
            u = torch.cat([u, u[:, -1:, :].expand(-1, pad, -1)], dim=1)
            xw = torch.cat([xw, xw[:, -1:, :, :].expand(-1, pad, -1, -1)], dim=1)

        P = self.P()  # (n,n), orthogonal
        Q = _rotation_block_diagonal(self.theta, self.n_state)  # (n,n), orthogonal
        A = P.T @ Q @ P  # similarity transform: preserves eigenvalues, changes basis

        Bu = self.B_in(u)  # (B,L,n)
        z = torch.zeros(Bsz, self.n_state, device=u.device, dtype=u.dtype)

        gate = None
        if self.damping_gate is not None:
            # Convert (B, L, C, W) into (B, nb, C, W) block summaries
            x_blk = xw.reshape(Bsz, nb, self.block_len, xw.shape[2], xw.shape[3]).mean(dim=2)
            gate = self.damping_gate(x_blk).to(u.dtype)  # (B,n)

        for t in range(u.shape[1]):
            z = z @ A.T + Bu[:, t, :]
            if gate is not None:
                z = z * gate
            z = self.norm(z)

        return self.head(z)


class ImplicitDiagJacobianOscillatorClassifier(nn.Module):
    """
    Invented "mathematician" upgrade:
      - Same oscillatory geometry A = Pᵀ Q P
      - Adds an *implicit* elementwise nonlinearity with diagonal Jacobian:
            z - α tanh(z) = rhs
        solved by a few Newton steps (elementwise, stable and fast).

    Why this matches recent trends:
      - "Diagonal Jacobian by design": the implicit equation's Jacobian wrt z is diagonal.
      - "Fixed-point / Newton view": we solve a (simple) implicit step, like DEER/ELK-like methods,
        but specialized to an elementwise nonlinearity so it's very cheap.

    Practical knobs:
      - alpha_max caps nonlinearity (keeps Newton stable)
      - newton_steps (1–3 is usually enough)
      - use_damp / r_min controls contraction and long-horizon stability
    """
    def __init__(
        self,
        d_in: int,
        C_in: int,
        n_state: int,
        n_classes: int,
        block_len: int = 8,
        eps: float = 1e-4,
        use_damp: bool = True,
        r_min: float = 0.98,
        newton_steps: int = 2,
        alpha_max: float = 0.95,
    ):
        super().__init__()
        assert n_state % 2 == 0, "n_state must be even (2×2 oscillator blocks)."
        self.n_state = int(n_state)
        self.block_len = int(block_len)
        self.newton_steps = int(newton_steps)
        self.alpha_max = float(alpha_max)

        self.P = CayleyOrthogonalParameterization(self.n_state, eps=eps, init_scale=2e-3)
        self.theta = nn.Parameter(torch.randn(self.n_state // 2) * 0.1)

        self.B_in = nn.Linear(int(d_in), self.n_state, bias=False)
        self.norm = nn.LayerNorm(self.n_state)
        self.head = nn.Linear(self.n_state, int(n_classes))

        self.damping_gate = BlockContextDampingGate(C_in=C_in, n_state=self.n_state, r_min=r_min) if use_damp else None

        # scalar α in (0, alpha_max)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

    def _solve_implicit(self, rhs: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Solve z - α tanh(z) = rhs with Newton's method (elementwise).

        f(z) = z - α tanh(z) - rhs
        f'(z) = 1 - α sech²(z)   (diagonal Jacobian)

        The update is:
            z ← z - f(z)/f'(z)
        """
        z = rhs
        for _ in range(self.newton_steps):
            th = torch.tanh(z)
            f = z - alpha * th - rhs
            sech2 = 1.0 - th * th
            J = 1.0 - alpha * sech2
            z = z - f / (J + 1e-6)
        return z

    def forward(self, u: torch.Tensor, xw: torch.Tensor) -> torch.Tensor:
        Bsz, L, _ = u.shape
        nb = (L + self.block_len - 1) // self.block_len
        pad = nb * self.block_len - L
        if pad:
            u = torch.cat([u, u[:, -1:, :].expand(-1, pad, -1)], dim=1)
            xw = torch.cat([xw, xw[:, -1:, :, :].expand(-1, pad, -1, -1)], dim=1)

        P = self.P()
        Q = _rotation_block_diagonal(self.theta, self.n_state)
        A = P.T @ Q @ P

        Bu = self.B_in(u)
        z = torch.zeros(Bsz, self.n_state, device=u.device, dtype=u.dtype)

        gate = None
        if self.damping_gate is not None:
            x_blk = xw.reshape(Bsz, nb, self.block_len, xw.shape[2], xw.shape[3]).mean(dim=2)
            gate = self.damping_gate(x_blk).to(u.dtype)  # (B,n)

        alpha = (self.alpha_max * torch.sigmoid(self.alpha_logit)).to(dtype=u.dtype, device=u.device)

        for t in range(u.shape[1]):
            rhs = z @ A.T + Bu[:, t, :]
            if gate is not None:
                rhs = rhs * gate
            z = self._solve_implicit(rhs, alpha)
            z = self.norm(z)

        return self.head(z)


# -----------------------------------------------------------------------------
# Backward-compatible aliases (keep original names so existing CLI flags work)
# -----------------------------------------------------------------------------
skew = skew_symmetric
cayley = cayley_transform
CayleyOrthogonal = CayleyOrthogonalParameterization
SimpleDampGate = BlockContextDampingGate
GOOSC_P_Classifier = OrthogonalOscillatorSSMClassifier
@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


@dataclass
class ClassifyConfig:
    device: str = "cpu"
    seed: int = 0
    window_len: int = 256
    step: int = 32
    n_state: int = 256
    block_len: int = 8
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    wd: float = 1e-4
    eps: float = 1e-4
    compile: bool = True
    compile_mode: str = "reduce-overhead"
    amp: bool = True
    num_workers: int = 4


def _maybe_enable_cuda_fastpaths(device: str) -> None:
    """Enable safe-ish perf knobs for NVIDIA GPUs."""
    if device != "cuda":
        return
    # TF32 is a common "free" speedup on Ampere+.
    try:
        # New (PyTorch 2.9+) preferred API.
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            # "tf32" enables TF32 on matmuls where applicable.
            torch.backends.cuda.matmul.fp32_precision = "tf32"
        else:
            # Back-compat for older PyTorch.
            torch.backends.cuda.matmul.allow_tf32 = True

        if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        else:
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def _maybe_compile(model: nn.Module, device: str, enabled: bool, mode: str) -> nn.Module:
    if (not enabled) or (device != "cuda"):
        return model
    # torch.compile is available in PyTorch 2.0+.
    if hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode=str(mode))
        except Exception:
            return model
    return model


def train_classifier(
    model: nn.Module,
    train_loader,
    test_loader,
    cfg: ClassifyConfig,
    mode: str,
    device: str,
) -> float:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    best = 0.0

    use_amp = bool(cfg.amp and device == "cuda")
    # PyTorch 2.8+ prefers torch.amp.* over torch.cuda.amp.*
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        _autocast = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        _autocast = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            if mode == "linoss":
                u, y = batch
                xw = None
            else:
                u, xw, y = batch

            if device == "cuda":
                u = u.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if xw is not None:
                    xw = xw.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with _autocast():
                if mode == "linoss":
                    logits = model(u)
                else:
                    logits = model(u, xw)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        if ep % 5 == 0 or ep == cfg.epochs:
            model.eval()
            tot = 0
            correct = 0
            with torch.no_grad():
                for batch in test_loader:
                    if mode == "linoss":
                        u, y = batch
                        xw = None
                    else:
                        u, xw, y = batch
                    if device == "cuda":
                        u = u.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        if xw is not None:
                            xw = xw.to(device, non_blocking=True)

                    with _autocast():
                        if mode == "linoss":
                            logits = model(u)
                        else:
                            logits = model(u, xw)
                    pred = logits.argmax(dim=1)
                    tot += int(y.numel())
                    correct += int((pred == y).sum().item())

            te = float(correct / max(1, tot))
            best = max(best, te)
    return float(best)




def _aeon_classification_dataset_names() -> list[str]:
    """
    Best-effort retrieval of available classification dataset names from aeon.

    Why this exists:
        aeon has had API changes across versions. Some versions expose a helper
        like `list_available_datasets`, others expose static name lists in
        `aeon.datasets.tsc_data_lists`.

    Returns:
        A list of dataset name strings (case-sensitive).
    """
    # Newer aeon (some versions)
    try:
        from aeon.datasets import list_available_datasets  # type: ignore
        return list_available_datasets(task="classification")
    except Exception:
        pass

    # Older aeon: name lists used in examples/docs (UCR/UEA archives)
    try:
        from aeon.datasets.tsc_data_lists import univariate, multivariate  # type: ignore
        # These are Python lists of strings in aeon<=0.7.x (and sometimes later).
        return sorted(set(list(univariate) + list(multivariate)))
    except Exception:
        pass

    # Last resort: no known listing API in this environment
    return []


def check_aeon_dataset_exists(dataset_name: str, task: str = "classification") -> None:
    """
    Fail early if a dataset name is not available in the local aeon registry.

    Notes:
        - On some aeon versions we can list datasets; on others we cannot.
        - If we cannot list datasets, we do *not* block execution here; instead
          the downstream `load_classification` call will raise a ValueError with
          details. This prevents false negatives due to API mismatch.

    Args:
        dataset_name: Dataset name string (case-sensitive).
        task: kept for signature compatibility; currently only "classification" is used.

    Raises:
        ValueError: if we can list datasets and dataset_name is not in the list.
    """
    names = _aeon_classification_dataset_names()
    if not names:
        # Cannot reliably pre-check on this aeon version.
        return

    if dataset_name not in names:
        low = dataset_name.lower()
        suggestions = [d for d in names if low in d.lower()]
        msg = (
            f"Dataset '{dataset_name}' not found in aeon (task='{task}').\n"
            f"Total available datasets (from aeon registry): {len(names)}"
        )
        if suggestions:
            msg += f"\nSuggestions: {suggestions[:10]}"
        raise ValueError(msg)


def run_classify_cli(ds_name: str, model_name: str, cfg: ClassifyConfig) -> pd.DataFrame:
    seed_all(cfg.seed)
    device = resolve_device(cfg.device)
    _maybe_enable_cuda_fastpaths(device)
    check_aeon_dataset_exists(ds_name, task="classification")

    Utr, Ute, Xwtr, Xwte, ytr, yte, C = load_uea_windows(ds_name, window_len=cfg.window_len, step=cfg.step, seed=cfg.seed)
    n_classes = int(ytr.max()) + 1
    d_in = int(Utr.shape[2])

    # Keep datasets on CPU and stream batches to GPU with pinned memory.
    # This avoids huge up-front H2D transfers and keeps the GPU fed.
    # NOTE: zero-copy strided window views from NumPy can be non-writeable; we never
    # mutate inputs, so it's safe to ignore PyTorch's warning about non-writeable arrays.
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"The given NumPy array is not writable, and PyTorch does not support non-writable tensors\.",
        category=UserWarning,
    )
    Utr_t = torch.from_numpy(Utr)
    Ute_t = torch.from_numpy(Ute)
    Xwtr_t = torch.from_numpy(Xwtr)
    Xwte_t = torch.from_numpy(Xwte)
    ytr_t = torch.from_numpy(ytr).long()
    yte_t = torch.from_numpy(yte).long()

    from torch.utils.data import DataLoader, TensorDataset

    pin = bool(device == "cuda")
    num_workers = int(cfg.num_workers) if device != "cpu" else 0

    if model_name == "linoss":
        train_ds = TensorDataset(Utr_t, ytr_t)
        test_ds = TensorDataset(Ute_t, yte_t)
    else:
        train_ds = TensorDataset(Utr_t, Xwtr_t, ytr_t)
        test_ds = TensorDataset(Ute_t, Xwte_t, yte_t)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=bool(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=bool(num_workers > 0),
    )

    if model_name == "linoss":
        model = LinOSSClassifier(d_in=d_in, n_state=cfg.n_state, n_classes=n_classes, eps=cfg.eps).to(device)
        model = _maybe_compile(model, device=device, enabled=cfg.compile, mode=cfg.compile_mode)
        acc = train_classifier(model, train_loader, test_loader, cfg, mode="linoss", device=device)
        pretty = "LinOSS"
    elif model_name == "go_osc_p":
        model = GOOSC_P_Classifier(d_in=d_in, C_in=C, n_state=cfg.n_state, n_classes=n_classes,
                                   block_len=cfg.block_len, eps=cfg.eps, use_damp=False).to(device)
        model = _maybe_compile(model, device=device, enabled=cfg.compile, mode=cfg.compile_mode)
        acc = train_classifier(model, train_loader, test_loader, cfg, mode="go", device=device)
        pretty = "GO-OSC+P"
    elif model_name == "go_osc_p_damp":
        model = GOOSC_P_Classifier(d_in=d_in, C_in=C, n_state=cfg.n_state, n_classes=n_classes,
                                   block_len=cfg.block_len, eps=cfg.eps, use_damp=True).to(device)
        model = _maybe_compile(model, device=device, enabled=cfg.compile, mode=cfg.compile_mode)
        acc = train_classifier(model, train_loader, test_loader, cfg, mode="go", device=device)
        pretty = "GO-OSC+P+DAMP"
    elif model_name == "go_osc_implicit":
        model = ImplicitDiagJacobianOscillatorClassifier(
            d_in=d_in,
            C_in=C,
            n_state=cfg.n_state,
            n_classes=n_classes,
            block_len=cfg.block_len,
            eps=cfg.eps,
            use_damp=True,
            r_min=0.98,
            newton_steps=2,
            alpha_max=0.95,
        ).to(device)
        model = _maybe_compile(model, device=device, enabled=cfg.compile, mode=cfg.compile_mode)
        acc = train_classifier(model, train_loader, test_loader, cfg, mode="go", device=device)
        pretty = "GO-OSC-Implicit(DiagJ)"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return pd.DataFrame([{
        "dataset": ds_name,
        "model": pretty,
        "accuracy": float(acc),
        "device": device,
        "seed": int(cfg.seed),
        "n_state": int(cfg.n_state),
        "block_len": int(cfg.block_len),
        "window_len": int(cfg.window_len),
        "step": int(cfg.step),
        "epochs": int(cfg.epochs),
    }])


# ============================================================
# Degradation (compact, classical-only outputs + placeholders)
# ============================================================

@dataclass
class DegradConfig:
    device: str = "cpu"
    seed: int = 0
    sensor_col: int = 0
    run_index: int = 0
    io_workers: int = 0  # 0 => auto


def run_degrad_ims(root: str, cfg: DegradConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runs = discover_runs_ims(root)
    run_index = int(cfg.run_index)
    if run_index < 0 or run_index >= len(runs):
        raise ValueError(f"run out of range. Found {len(runs)} runs")
    _, files = runs[run_index]
    feats = _parallel_features_numeric(files, col=cfg.sensor_col, max_workers=(cfg.io_workers or None))
    df = pd.DataFrame(feats)
    df.insert(0, "t_idx", np.arange(len(df), dtype=np.int64))
    df_det = pd.DataFrame([{"score": "rms", "t_detect": None, "thr": None}])
    return df, df_det


def run_degrad_pronostia(root: str, cfg: DegradConfig, mat_key: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runs = discover_runs_pronostia(root)
    run_index = int(cfg.run_index)
    if run_index < 0 or run_index >= len(runs):
        raise ValueError(f"run out of range. Found {len(runs)} runs")
    _, files = runs[run_index]
    # .mat loading tends to be more CPU-parsing heavy; keep default as sequential to avoid overhead.
    # You can parallelize similarly if your storage can handle it.
    sig = [_read_pronostia_mat(f, key=mat_key, col=cfg.sensor_col) for f in files]
    df = pd.DataFrame([{"t_idx": i, "rms": rms(x), "kurtosis": kurtosis(x), "crest": crest_factor(x)} for i, x in enumerate(sig)])
    df_det = pd.DataFrame([{"score": "rms", "t_detect": None, "thr": None}])
    return df, df_det


def run_degrad_xjtu(root: str, cfg: DegradConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runs = discover_runs_xjtu(root)
    run_index = int(cfg.run_index)
    if run_index < 0 or run_index >= len(runs):
        raise ValueError(f"run out of range. Found {len(runs)} runs")
    _, files = runs[run_index]
    feats = _parallel_features_numeric(files, col=cfg.sensor_col, max_workers=(cfg.io_workers or None))
    df = pd.DataFrame(feats)
    df.insert(0, "t_idx", np.arange(len(df), dtype=np.int64))
    df_det = pd.DataFrame([{"score": "rms", "t_detect": None, "thr": None}])
    return df, df_det


# ============================================================
# CLI
# ============================================================

def _cli():
    ap = argparse.ArgumentParser(description="OSCDYN benchmark (classification + degradation)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # degradation
    def add_degrad(sp):
        sp.add_argument("--root", required=True, type=str)
        sp.add_argument("--run", type=int, default=0)
        sp.add_argument("--sensor-col", type=int, default=0)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--device", type=str, default="cpu")
        sp.add_argument("--io-workers", type=int, default=0, help="Thread workers for txt/csv reads (0=auto)")
        sp.add_argument("--out", type=str, default=None)
        sp.add_argument("--out-det", type=str, default=None)

    sp_ims = sub.add_parser("ims")
    add_degrad(sp_ims)

    sp_pro = sub.add_parser("pronostia")
    add_degrad(sp_pro)
    sp_pro.add_argument("--mat-key", type=str, default=None)

    sp_xjtu = sub.add_parser("xjtu")
    add_degrad(sp_xjtu)

    # classification
    sp_cls = sub.add_parser("classify")
    sp_cls.add_argument("--dataset", required=True, type=str)
    sp_cls.add_argument("--list-datasets", action="store_true", help="List available aeon classification datasets and exit")
    sp_cls.add_argument("--model", default="go_osc_p_damp", choices=["linoss", "go_osc_p", "go_osc_p_damp", "go_osc_implicit"])
    sp_cls.add_argument("--device", default="cpu", type=str)
    sp_cls.add_argument("--seed", default=0, type=int)
    sp_cls.add_argument("--n-state", default=256, type=int)
    sp_cls.add_argument("--block-len", default=8, type=int)
    sp_cls.add_argument("--window-len", default=256, type=int)
    sp_cls.add_argument("--step", default=32, type=int)
    sp_cls.add_argument("--epochs", default=100, type=int)
    sp_cls.add_argument("--batch-size", default=256, type=int)
    sp_cls.add_argument("--lr", default=1e-3, type=float)
    sp_cls.add_argument("--wd", default=1e-4, type=float)
    sp_cls.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    sp_cls.add_argument("--compile-mode", default="reduce-overhead", type=str, help="torch.compile mode")
    sp_cls.add_argument("--no-amp", action="store_true", help="Disable AMP autocast + GradScaler")
    sp_cls.add_argument("--num-workers", default=4, type=int, help="DataLoader workers (GPU only; CPU uses 0)")
    sp_cls.add_argument("--out", default=None, type=str)

    args = ap.parse_args()

    
    if getattr(args, "list_datasets", False):
        datasets = _aeon_classification_dataset_names()
        if not datasets:
            print("Could not list datasets from aeon in this environment.")
            print("Try updating aeon, or rely on aeon.datasets.load_classification(name=...) errors.")
            return

        datasets = sorted(datasets)
        print("\nAvailable aeon classification datasets:")
        print("---------------------------------------")
        for d in datasets:
            print(" -", d)
        print(f"\nTotal: {len(datasets)} datasets")
        return

    if args.cmd == "classify":
        cfg = ClassifyConfig(
            device=args.device, seed=args.seed, n_state=args.n_state, block_len=args.block_len,
            window_len=args.window_len, step=args.step, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, wd=args.wd,
            compile=not bool(args.no_compile), compile_mode=str(args.compile_mode),
            amp=not bool(args.no_amp), num_workers=int(args.num_workers),
        )
        df = run_classify_cli(args.dataset, args.model, cfg)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.out, index=False)
            print(f"Wrote -> {args.out}")
        else:
            print(df.to_string(index=False))
        return

    # degradation
    cfgd = DegradConfig(
        device=args.device,
        seed=args.seed,
        sensor_col=args.sensor_col,
        run_index=args.run,
        io_workers=int(getattr(args, "io_workers", 0) or 0),
    )
    if args.cmd == "ims":
        df, df_det = run_degrad_ims(args.root, cfgd)
    elif args.cmd == "pronostia":
        df, df_det = run_degrad_pronostia(args.root, cfgd, mat_key=args.mat_key)
    else:
        df, df_det = run_degrad_xjtu(args.root, cfgd)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote df -> {args.out}")
    else:
        print(df.head().to_string(index=False))

    if args.out_det:
        Path(args.out_det).parent.mkdir(parents=True, exist_ok=True)
        df_det.to_csv(args.out_det, index=False)
        print(f"Wrote det -> {args.out_det}")
    else:
        print(df_det.to_string(index=False))


if __name__ == "__main__":
    _cli()
