# GO-OSC & VASH  
**Geometry-Aware Oscillatory Representation for Early Degradation Detection**

**Brand:** Datar Consulting  
**Author:** Vashista Nobaub  
**Contact:** labs@datar.fr  

This repository contains reference code accompanying the paper:

> **GO-OSC and VASH: Geometry-Aware Representation Learning for Early Degradation Detection**

It provides:
- a compact **benchmark + CLI** for classification and degradation experiments, and  
- a **feature / indicator library (VASH)** for classical and geometry-aware health monitoring.

The focus is early, phase-dominated degradation in oscillatory systems (e.g. bearings, rotating machinery).

---

## 1. Key Ideas

### GO-OSC (Geometry-Aware Oscillatory Representation)
GO-OSC models signals as latent oscillatory dynamics with:
- explicit frequency and damping structure,
- a **canonical parameterization** (fixed gauge),
- stable, comparable latent coordinates across windows.

This makes subtle phase and coherence changes detectable before energy changes.

### VASH (Vibration-Aware Structural Health indicators)
VASH is a set of invariant geometric indicators derived from GO-OSC outputs, including:
- **PCC** – Phase Coherence Collapse  
- **FWR** – Frequency Wander Rate  
- **DDI** – Damping Drift Integral  
- **GSI** – Geometric State Indicator  
- **MLL** – Mode-Locking Loss  
- **LQF** – Latent Q-Factor proxy  

These are designed to respond earlier than RMS or spectral power.

---

## 2. Repository Structure

```
.
├── oscdyn_degradation_benchmark_*.py   # Benchmark + CLI (classification & degradation)
├── vash_indicators_paper.py            # VASH feature & indicator library
├── README.md
```

---

## 3. Requirements

Minimal (CPU-only):

```
pip install numpy pandas torch scikit-learn
```

Optional:
```
pip install aeon
pip install scipy
pip install pywt PyEMD
```

---

## 4. Benchmark Script  
`oscdyn_degradation_benchmark_*.py`

### Purpose
A compact experimentation script supporting:
- **Time-series classification** (UEA/UCR via `aeon`)
- **Run-to-failure / degradation datasets** (IMS, PRONOSTIA, XJTU-SY)
- GO-OSC-style oscillatory classifiers for research comparison

---

### 4.1 Classification (UEA / UCR)

```
python oscdyn_degradation_benchmark.py classify \
  --dataset EthanolConcentration \
  --model go_osc_p_damp \
  --device cpu
```

Available models:
- `linoss`
- `go_osc_p`
- `go_osc_p_damp`
- `go_osc_implicit`

---

### 4.2 Degradation / Run-to-Failure

IMS:
```
python oscdyn_degradation_benchmark.py ims --root /path/IMS --run 0
```

PRONOSTIA:
```
python oscdyn_degradation_benchmark.py pronostia --root /path/PRONOSTIA --run 1
```

XJTU-SY:
```
python oscdyn_degradation_benchmark.py xjtu --root /path/XJTU --run 0
```

---

## 5. VASH Indicator Library  
`vash_indicators_paper.py`

### Capabilities
- Classical time, frequency, and time–frequency indicators
- Nonlinear and complexity metrics
- GO-OSC / VASH-native geometric indicators
- Health Index construction with baseline normalization

---

## 6. Example Usage

```python
from vash_indicators_paper import FeatureBank, HealthIndex

fb = FeatureBank(fs=25600)
features = fb.compute_window(signal)

hi = HealthIndex(weights={"GSI":1.0,"PCC":1.0,"FWR":0.5,"DDI":0.5})
hi.fit_baseline(healthy_feature_dicts)

score = hi.score(features)
```

---

## 7. Scope & Limitations

- Targets **early, phase-only degradation**
- Assumes oscillatory dynamics and local stationarity
- Not optimized for late-stage, high-energy faults

---

## 8. Intended Audience

Researchers and practitioners in:
- condition monitoring
- time-series machine learning
- early fault detection

---

## 9. Disclaimer

Research code only.  
Not certified for safety-critical deployment.
---

## 10. Google Colab Notebook

A companion **Google Colab notebook** is provided to reproduce key figures, indicators, and analysis from the paper without requiring local setup.

**Notebook:** `GOOSC_VASH_PAPER_FIGURES.ipynb`

### What the notebook demonstrates
- Loading and preprocessing oscillatory vibration signals  
- Computing classical vibration indicators (RMS, kurtosis, spectral features)  
- Computing **GO-OSC / VASH geometric indicators** (PCC, GSI, FWR, DDI, MLL, LQF)  
- Visualizing early degradation effects where energy-based metrics fail  
- Reproducing representative plots and figures used in the paper  

### Intended use
The notebook is designed for:
- reviewers and readers of the paper,
- rapid experimentation and inspection of indicators,
- educational and presentation purposes.

It runs entirely on **CPU** in Google Colab and installs only lightweight dependencies.

### Typical workflow in Colab
1. Open the notebook in Google Colab  
2. Run the setup cell to install dependencies  
3. Execute cells sequentially to reproduce figures and indicator trends  
4. Modify signal sources or parameters to explore sensitivity  

No training is required; the notebook focuses on **analysis and interpretation**, not model fitting.

---

