# RTTD: Randomized Tensor-Train Truncation on GPUs for Quantum-Inspired CFD

This repository contains **RTTD**, a GPU-accelerated randomized Tensor-Train (TT/MPS) truncation routine designed
as a drop-in replacement for deterministic SVD-based truncation inside TT/MPO workflows used in quantum-inspired
computational fluid dynamics (QIS–CFD).

RTTD shifts the truncation workload toward **GEMM + thin QR** operations while preserving the canonical constraints
required by TT/MPS sweeps. The method is evaluated on 2D benchmarks (decaying jet and Taylor–Green vortex).

## Paper
- Title: *Randomized Tensor-Train Truncation on GPUs for Quantum-Inspired Computational Fluid Dynamics*
- Authors: Kiet Tuan Pham, Jinsung Kim

## What’s in this repo
- `RTTD_module/` : C++/CUDA implementation of RTTD (cuBLAS + cuSOLVER)
- `2D_DNS/` : DNS Solution
- `2D_QIS/` : QIS–CFD pipeline integration (RTTD as drop-in truncation for contract+decompose-style steps)
- `Plot/` : Reproduction scripts for:
  - Taylor–Green vortex (TGV)
  - Decaying jet (DJ)

## Quick start

### 1) Create environment
```bash
conda env create -f environment.yml
conda activate cutensor
```

### 2) Build/install
```bash
cd RTTD_module/
make -j8
make install
```

### 3) Run simulation
```bash
cd 2D_QIS
python simulation.py
```

### 4) Reproduce a paper result
```bash
cd Plot
```

## Reproducing the paper

### Default RTTD hyperparameters
- Oversampling `p = 5`
- Power iterations `q = 1`

### Reported environment
- GPU: NVIDIA GeForce RTX 4090 (24GB)
- CUDA: 12.6
- Libraries: cuBLAS 12, cuSOLVER 11, cuQuantum 24.08, CuPy 13.6
- Python: 3.12

> Notes:
> - If you use different GPUs or CUDA versions, performance and numerical results may vary slightly.
> - For the most faithful reproduction, use the same library versions as reported above.

## Benchmarks

### Taylor–Green vortex (TGV)
- Re = 1e3, n = 10, χ ∈ {32, 64}
- Diagnostics: energy, divergence, relative L2 error vs analytic decay

### Decaying jet (DJ)
- Re ∈ {1e5, 2e5}, n ∈ {10, 11, 12}, χ ∈ {16, 32, 64, 100, 128}
- Diagnostics: fidelity vs DNS, energy, divergence, spectrum

### Third-party code
This repository includes code adapted from:
- **QIFS - Quantum-Inspired Fluid Simulations** (Leonhard Hoelscher), licensed under **Apache-2.0**

## Citation
If you use RTTD, please cite
