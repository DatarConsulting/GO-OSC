"""
vash_indicators.py

VASH / Datar Consulting
----------------------
A practical, industrial feature bank for vibration degradation + early anomaly.

This module computes:
- Classical time-domain indicators (RMS, kurtosis, crest, etc.)
- Envelope/Hilbert indicators
- Frequency-domain indicators (centroid, bandwidth, entropy, band energies, flux)
- Time-frequency indicators (STFT-based spectral kurtosis proxy; optional wavelets)
- Nonlinear/complexity indicators (SampEn, PermEn, LZ, DFA, Hurst, fractal dims)
- Online change-detection utilities (EWMA, CUSUM, Page-Hinkley)
- GO-OSC/VASH-native indicators (geo, damping, omega) as inputs + derived indicators:
  PCC, DDI, GSI, FWR, MLL, LQF
- A HealthIndex class that builds a single scalar HI(t) from features + GO-OSC signals,
  using baseline normalization learned on a healthy segment.
 
Dependencies:
- Required: numpy
- Optional: scipy (hilbert, stft); if absent, envelope & STFT features are skipped.
- Optional: pywt (wavelets) and PyEMD (EMD) are supported if installed.

CPU-friendly, streaming-friendly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.signal import hilbert, stft  # type: ignore
except Exception:
    hilbert = None
    stft = None

try:
    import pywt  # type: ignore
except Exception:
    pywt = None

try:
    from PyEMD import EMD  # type: ignore
except Exception:
    EMD = None

EPS = 1e-12


# ---------------- Time-domain ----------------

def mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x))

def std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.std(x) + EPS)

def var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.var(x) + EPS)

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + EPS))

def peak_to_peak(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(x) - np.min(x))

def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = np.mean(x)
    s = np.std(x) + EPS
    z = (x - m) / s
    return float(np.mean(z**3))

def kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = np.mean(x)
    s2 = np.var(x) + EPS
    z = (x - m) / math.sqrt(s2)
    return float(np.mean(z**4))

def crest_factor(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(np.abs(x)) / (rms(x) + EPS))

def impulse_factor(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(np.abs(x)) / (np.mean(np.abs(x)) + EPS))

def shape_factor(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(rms(x) / (np.mean(np.abs(x)) + EPS))

def clearance_factor(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(np.abs(x)) / ((np.mean(np.sqrt(np.abs(x))) + EPS) ** 2))


# ---------------- Envelope / Hilbert ----------------

def envelope(x: np.ndarray) -> Optional[np.ndarray]:
    if hilbert is None:
        return None
    x = np.asarray(x, dtype=np.float64)
    return np.abs(hilbert(x))

def envelope_rms(x: np.ndarray) -> float:
    env = envelope(x)
    if env is None:
        return float("nan")
    return rms(env)

def envelope_kurtosis(x: np.ndarray) -> float:
    env = envelope(x)
    if env is None:
        return float("nan")
    return kurtosis(env)


# ---------------- TKEO ----------------

def tkeo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return np.zeros_like(x)
    y = np.zeros_like(x)
    y[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    y[0] = y[1]
    y[-1] = y[-2]
    return y

def tkeo_mean(x: np.ndarray) -> float:
    return float(np.mean(np.abs(tkeo(x))))

def tkeo_std(x: np.ndarray) -> float:
    return float(np.std(tkeo(x)) + EPS)


# ---------------- Frequency / STFT ----------------

def _rfft_psd(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n <= 1:
        return np.array([0.0]), np.array([0.0])
    X = np.fft.rfft(x * np.hanning(n))
    p = (np.abs(X) ** 2) / (n * fs + EPS)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    return f, p

def spectral_centroid(x: np.ndarray, fs: float) -> float:
    f, p = _rfft_psd(x, fs)
    s = float(np.sum(p) + EPS)
    return float(np.sum(f * p) / s)

def spectral_bandwidth(x: np.ndarray, fs: float) -> float:
    f, p = _rfft_psd(x, fs)
    c = spectral_centroid(x, fs)
    s = float(np.sum(p) + EPS)
    return float(np.sqrt(np.sum(((f - c) ** 2) * p) / s))

def spectral_entropy(x: np.ndarray, fs: float) -> float:
    _, p = _rfft_psd(x, fs)
    p = p.astype(np.float64)
    p = p / (np.sum(p) + EPS)
    return float(-np.sum(p * np.log(p + EPS)))

def band_energy_ratios(x: np.ndarray, fs: float, bands: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    f, p = _rfft_psd(x, fs)
    total = float(np.sum(p) + EPS)
    out: Dict[str, float] = {}
    for (lo, hi) in bands:
        m = (f >= lo) & (f < hi)
        e = float(np.sum(p[m]))
        out[f"band_energy_{lo:g}_{hi:g}"] = e
        out[f"band_ratio_{lo:g}_{hi:g}"] = e / total
    return out

def spectral_flux(x: np.ndarray, fs: float, prev_psd: Optional[np.ndarray]) -> Tuple[float, np.ndarray]:
    _, p = _rfft_psd(x, fs)
    p = p / (np.sum(p) + EPS)
    if prev_psd is None or prev_psd.shape != p.shape:
        return float("nan"), p
    diff = p - prev_psd
    return float(np.sqrt(np.sum(diff * diff))), p

def stft_spectral_kurtosis_proxy(x: np.ndarray, fs: float, nperseg: int = 1024, noverlap: int = 512) -> float:
    if stft is None:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    _, _, Z = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
    mag = np.abs(Z) + EPS  # (F,T)
    m = mag.mean(axis=1, keepdims=True)
    s = mag.std(axis=1, keepdims=True) + EPS
    z = (mag - m) / s
    k = np.mean(z**4, axis=1)
    return float(np.mean(k))


# ---------------- Optional wavelets / EMD ----------------

def cwt_energy_entropy(x: np.ndarray, wavelet: str = "morl", scales: Optional[np.ndarray] = None) -> float:
    if pywt is None:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    if scales is None:
        scales = np.arange(1, 64)
    coef, _ = pywt.cwt(x, scales, wavelet)
    e = (np.abs(coef) ** 2).reshape(-1)
    e = e / (np.sum(e) + EPS)
    return float(-np.sum(e * np.log(e + EPS)))

def wavelet_packet_energy_entropy(x: np.ndarray, wavelet: str = "db4", level: int = 4) -> float:
    if pywt is None:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    wp = pywt.WaveletPacket(data=x, wavelet=wavelet, maxlevel=level)
    nodes = wp.get_level(level, order="freq")
    energies = np.array([np.sum(np.square(n.data)) for n in nodes], dtype=np.float64)
    energies = energies / (np.sum(energies) + EPS)
    return float(-np.sum(energies * np.log(energies + EPS)))

def emd_imf_energy_kurtosis(x: np.ndarray, max_imfs: int = 6) -> Dict[str, float]:
    if EMD is None:
        return {"emd_available": 0.0}
    x = np.asarray(x, dtype=np.float64)
    emd = EMD()
    imfs = emd.emd(x)
    out: Dict[str, float] = {"emd_available": 1.0}
    for i in range(min(max_imfs, imfs.shape[0])):
        imf = imfs[i]
        out[f"imf{i+1}_energy"] = float(np.mean(imf**2))
        out[f"imf{i+1}_kurtosis"] = kurtosis(imf)
    return out


# ---------------- Complexity / nonlinear ----------------

def sample_entropy(x: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < m + 2:
        return float("nan")
    if r is None:
        r = 0.2 * (np.std(x) + EPS)

    def _phi(mm: int) -> float:
        count = 0
        total = 0
        for i in range(n - mm):
            xi = x[i:i+mm]
            for j in range(i + 1, n - mm + 1):
                xj = x[j:j+mm]
                total += 1
                if np.max(np.abs(xi - xj)) <= r:
                    count += 1
        return (count + EPS) / (total + EPS)

    return float(-np.log(_phi(m + 1) / _phi(m)))

def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < (order - 1) * delay + 1:
        return float("nan")
    patterns: Dict[Tuple[int, ...], int] = {}
    for i in range(n - (order - 1) * delay):
        window = x[i:(i + order * delay):delay]
        key = tuple(np.argsort(window))
        patterns[key] = patterns.get(key, 0) + 1
    counts = np.array(list(patterns.values()), dtype=np.float64)
    p = counts / (np.sum(counts) + EPS)
    return float(-np.sum(p * np.log(p + EPS)))

def lempel_ziv_complexity(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    med = np.median(x)
    s = (x > med).astype(np.uint8)
    i, k, l = 0, 1, 1
    c = 1
    n = len(s)
    while True:
        if i + k >= n or l + k >= n:
            c += 1
            break
        if np.array_equal(s[i:i+k], s[l:l+k]):
            k += 1
        else:
            if k > 1:
                i += 1
                k = 1
            else:
                c += 1
                l += 1
                i = 0
        if l >= n:
            break
    return float(c)

def dfa_alpha(x: np.ndarray, min_scale: int = 16, max_scale: int = 256, n_scales: int = 8) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < max_scale * 2:
        return float("nan")
    y = np.cumsum(x - np.mean(x))
    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int))
    F = []
    for s in scales:
        if s < 4:
            continue
        nseg = n // s
        if nseg < 2:
            continue
        rms_seg = []
        for v in range(nseg):
            seg = y[v*s:(v+1)*s]
            t = np.arange(s)
            p = np.polyfit(t, seg, 1)
            trend = np.polyval(p, t)
            rms_seg.append(np.sqrt(np.mean((seg - trend)**2) + EPS))
        F.append(np.mean(rms_seg))
    if len(F) < 2:
        return float("nan")
    log_s = np.log(scales[:len(F)] + EPS)
    log_F = np.log(np.array(F) + EPS)
    a, _ = np.polyfit(log_s, log_F, 1)
    return float(a)

def hurst_exponent_rs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 64:
        return float("nan")
    y = x - np.mean(x)
    z = np.cumsum(y)
    R = np.max(z) - np.min(z)
    S = np.std(y) + EPS
    return float(np.log((R / S) + EPS) / np.log(n))

def katz_fractal_dimension(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        return float("nan")
    L = np.sum(np.abs(np.diff(x)))
    d = np.max(np.abs(x - x[0])) + EPS
    return float(np.log10(n) / (np.log10(d / (L + EPS)) + np.log10(n)))

def higuchi_fractal_dimension(x: np.ndarray, kmax: int = 8) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2 * kmax:
        return float("nan")
    Lk = []
    k_list = range(1, kmax + 1)
    for k in k_list:
        Lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if idx.size < 2:
                continue
            lm = np.sum(np.abs(np.diff(x[idx])))
            norm = (n - 1) / (k * (idx.size - 1) + EPS)
            Lm.append(lm * norm)
        if Lm:
            Lk.append(np.mean(Lm))
    if len(Lk) < 2:
        return float("nan")
    logk = np.log(1.0 / np.array(list(k_list[:len(Lk)]), dtype=np.float64))
    logL = np.log(np.array(Lk, dtype=np.float64) + EPS)
    slope, _ = np.polyfit(logk, logL, 1)
    return float(slope)


# ---------------- Online change detection ----------------

@dataclass
class EWMA:
    alpha: float = 0.1
    value: Optional[float] = None
    def update(self, x: float) -> float:
        if self.value is None or not np.isfinite(self.value):
            self.value = float(x)
        else:
            self.value = float(self.alpha * x + (1 - self.alpha) * self.value)
        return float(self.value)

@dataclass
class CUSUM:
    k: float = 0.5
    h: float = 5.0
    pos: float = 0.0
    neg: float = 0.0
    mu0: float = 0.0
    def update(self, x: float) -> Dict[str, Any]:
        s = x - self.mu0
        self.pos = max(0.0, self.pos + s - self.k)
        self.neg = min(0.0, self.neg + s + self.k)
        alarm = (self.pos > self.h) or (abs(self.neg) > self.h)
        return {"pos": self.pos, "neg": self.neg, "alarm": alarm}

@dataclass
class PageHinkley:
    delta: float = 0.005
    lam: float = 50.0
    mean: float = 0.0
    mT: float = 0.0
    MT: float = 0.0
    t: int = 0
    def update(self, x: float) -> Dict[str, Any]:
        self.t += 1
        self.mean = self.mean + (x - self.mean) / self.t
        self.mT += x - self.mean - self.delta
        self.MT = min(self.MT, self.mT)
        alarm = (self.mT - self.MT) > self.lam
        return {"mean": self.mean, "mT": self.mT, "MT": self.MT, "alarm": alarm}


# ---------------- VASH-native (GO-OSC-derived) ----------------

def pcc_from_phase_increments(dphi: np.ndarray) -> float:
    dphi = np.asarray(dphi, dtype=np.float64).reshape(-1)
    if dphi.size == 0:
        return float("nan")
    z = np.exp(1j * dphi)
    return float(1.0 - np.abs(np.mean(z)))

def ddi_increment(damp: float, damp0: float) -> float:
    if not np.isfinite(damp) or not np.isfinite(damp0):
        return float("nan")
    return float(max(0.0, damp0 - damp))

def gsi_from_geo(geo: float, mu: float, sig: float) -> float:
    return float((geo - mu) / (sig + EPS))

def fwr_from_omega(omega: np.ndarray, omega_prev: Optional[np.ndarray]) -> float:
    omega = np.asarray(omega, dtype=np.float64).reshape(-1)
    if omega_prev is None:
        return float("nan")
    omega_prev = np.asarray(omega_prev, dtype=np.float64).reshape(-1)
    if omega.shape != omega_prev.shape:
        return float("nan")
    return float(np.sum(np.abs(omega - omega_prev)))

def mll_from_omega(omega: np.ndarray) -> float:
    omega = np.asarray(omega, dtype=np.float64).reshape(-1)
    omega = omega[np.isfinite(omega)]
    if omega.size < 2:
        return float("nan")
    ratios = []
    for i in range(omega.size):
        for j in range(i + 1, omega.size):
            ratios.append(omega[i] / (omega[j] + EPS))
    return float(np.var(ratios) + EPS)

def lqf_from_omega_damp(omega: np.ndarray, damp: float) -> float:
    omega = np.asarray(omega, dtype=np.float64).reshape(-1)
    if omega.size == 0 or not np.isfinite(damp):
        return float("nan")
    return float(np.linalg.norm(omega) / (1.0 - damp + EPS))



# ---- Public wrapper names used by notebooks ----
# These aliases keep the notebook API stable.

def compute_pcc(dphi: np.ndarray) -> float:
    """Alias for PCC."""
    return pcc_from_phase_increments(dphi)

def compute_gsi(geo: float, mu: float = 0.0, sig: float = 1.0) -> float:
    """If mu/sig provided, returns standardized geo surprise; else returns geo itself."""
    # Notebook typically passes only geo; in that case return geo (raw) and HI will standardize via baseline.
    if sig is None or sig == 0.0:
        sig = 1.0
    if mu == 0.0 and sig == 1.0:
        return float(geo)
    return gsi_from_geo(geo, mu, sig)

def compute_fwr(omega: np.ndarray, omega_prev: Optional[np.ndarray]) -> float:
    """Alias for frequency wander."""
    return fwr_from_omega(omega, omega_prev)

# ---------------- FeatureBank ----------------

@dataclass
class FeatureBank:
    fs: float
    band_edges: Sequence[Tuple[float, float]] = ((0.0, 500.0), (500.0, 2000.0), (2000.0, 8000.0))
    stft_nperseg: int = 1024
    stft_noverlap: int = 512
    downsample_for_complexity: int = 10
    enable_wavelets: bool = False
    enable_emd: bool = False

    def compute_window(self, x: np.ndarray, prev: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        out: Dict[str, float] = {}

        # Time-domain
        out["mean"] = mean(x)
        out["std"] = std(x)
        out["var"] = var(x)
        out["rms"] = rms(x)
        out["ptp"] = peak_to_peak(x)
        out["skew"] = skewness(x)
        out["kurtosis"] = kurtosis(x)
        out["crest"] = crest_factor(x)
        out["impulse"] = impulse_factor(x)
        out["shape"] = shape_factor(x)
        out["clearance"] = clearance_factor(x)

        # Envelope
        out["env_rms"] = envelope_rms(x)
        out["env_kurtosis"] = envelope_kurtosis(x)

        # TKEO
        out["tkeo_mean"] = tkeo_mean(x)
        out["tkeo_std"] = tkeo_std(x)

        # Frequency
        out["spec_centroid"] = spectral_centroid(x, self.fs)
        out["spec_bandwidth"] = spectral_bandwidth(x, self.fs)
        out["spec_entropy"] = spectral_entropy(x, self.fs)
        out.update(band_energy_ratios(x, self.fs, self.band_edges))

        # Spectral flux
        prev_psd = None if prev is None else prev.get("psd")
        flux, psd = spectral_flux(x, self.fs, prev_psd)
        out["spec_flux"] = flux
        if prev is not None:
            prev["psd"] = psd

        # STFT spectral kurtosis proxy
        out["stft_spectral_kurtosis"] = stft_spectral_kurtosis_proxy(
            x, self.fs, nperseg=self.stft_nperseg, noverlap=self.stft_noverlap
        )

        # Optional wavelets
        if self.enable_wavelets:
            out["cwt_energy_entropy"] = cwt_energy_entropy(x)
            out["wpt_energy_entropy"] = wavelet_packet_energy_entropy(x)
        else:
            out["cwt_energy_entropy"] = float("nan")
            out["wpt_energy_entropy"] = float("nan")

        # Optional EMD
        if self.enable_emd:
            out.update(emd_imf_energy_kurtosis(x))
        else:
            out["emd_available"] = 0.0

        # Complexity (decimated)
        ds = max(1, int(self.downsample_for_complexity))
        xd = x[::ds]
        out["sampen"] = sample_entropy(xd, m=2, r=None)
        out["permen"] = permutation_entropy(xd, order=3, delay=1)
        out["lz"] = lempel_ziv_complexity(xd)
        out["dfa_alpha"] = dfa_alpha(xd, min_scale=8, max_scale=64, n_scales=6)
        out["hurst"] = hurst_exponent_rs(xd)
        out["fd_katz"] = katz_fractal_dimension(xd)
        out["fd_higuchi"] = higuchi_fractal_dimension(xd, kmax=8)
        return out


# ---------------- HealthIndex ----------------

@dataclass
class HealthIndex:
    weights: Dict[str, float]
    baseline: Optional[Dict[str, Tuple[float, float]]] = None
    damp0: Optional[float] = None
    ddi: float = 0.0

    def fit_baseline(self, feats_list: Sequence[Dict[str, Any]], geo_key: str = "geo", damp_key: str = "damp_mean") -> None:
        keys = set()
        for d in feats_list:
            keys |= set(d.keys())
        baseline: Dict[str, Tuple[float, float]] = {}
        for k in keys:
            vals = np.array([d.get(k, np.nan) for d in feats_list], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            baseline[k] = (float(vals.mean()), float(vals.std() + EPS))
        self.baseline = baseline
        self.damp0 = baseline.get(damp_key, (np.nan, np.nan))[0] if damp_key in baseline else None
        self.ddi = 0.0

    def _z(self, k: str, v: float) -> float:
        if self.baseline is None or k not in self.baseline or not np.isfinite(v):
            return float("nan")
        mu, sd = self.baseline[k]
        return float((v - mu) / (sd + EPS))

    def score(self, feats: Dict[str, Any], prev_feats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.baseline is None:
            raise RuntimeError("Call fit_baseline() before score().")

        out: Dict[str, Any] = {}
        geo = float(feats.get("geo", np.nan))
        damp = float(feats.get("damp_mean", np.nan))

        # Derived indicators
        if np.isfinite(geo) and "geo" in self.baseline:
            mu, sd = self.baseline["geo"]
            out["GSI"] = gsi_from_geo(geo, mu, sd)
        else:
            out["GSI"] = float("nan")

        out["damp_loss"] = float(1.0 - damp) if np.isfinite(damp) else float("nan")

        if self.damp0 is not None and np.isfinite(damp):
            self.ddi += ddi_increment(damp, float(self.damp0))
            out["DDI"] = float(self.ddi)
        else:
            out["DDI"] = float("nan")

        omega = feats.get("omega", None)
        omega_prev = None if prev_feats is None else prev_feats.get("omega", None)
        if omega is not None:
            out["MLL"] = mll_from_omega(np.asarray(omega))
            out["LQF"] = lqf_from_omega_damp(np.asarray(omega), damp) if np.isfinite(damp) else float("nan")
            out["FWR"] = fwr_from_omega(np.asarray(omega), None if omega_prev is None else np.asarray(omega_prev))
        else:
            out["MLL"] = out["LQF"] = out["FWR"] = float("nan")

        dphi = feats.get("dphi", None)
        out["PCC"] = pcc_from_phase_increments(np.asarray(dphi)) if dphi is not None else float("nan")

        # Copy scalar features
        for k, v in feats.items():
            if isinstance(v, (int, float, np.floating)):
                out[k] = float(v)

        # Health Index
        terms = []
        for k, w in self.weights.items():
            w = float(w)
            if w == 0.0:
                continue
            if k in out:
                v = float(out[k])
                if k in ("GSI", "damp_loss", "FWR", "PCC", "MLL", "LQF", "DDI"):
                    s = v
                else:
                    s = self._z(k, v)
                if np.isfinite(s):
                    terms.append(w * s)
        out["HI"] = float(np.sum(terms)) if terms else float("nan")

        # Basic alerts
        out["alerts"] = {
            "hi_alarm": bool(np.isfinite(out["HI"]) and out["HI"] > 3.0),
            "geo_alarm": bool(np.isfinite(out.get("GSI", np.nan)) and out["GSI"] > 3.0),
            "damp_alarm": bool(np.isfinite(out.get("damp_loss", np.nan)) and out["damp_loss"] > 0.02),
        }
        return out
