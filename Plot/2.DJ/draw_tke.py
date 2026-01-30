#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.size": 20,
    "font.family": "serif",
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.compression": 9,
})


def load_mps_cores(fname: str):
    data = np.load(fname, allow_pickle=True)
    if getattr(data, "ndim", None) == 0:
        data = data.item()
    cores = data if isinstance(data, list) else list(data)
    return [np.asarray(c, dtype=np.float64) for c in cores]


def reconstruct_2d_field(cores, n_bits: int):
    N = 2 ** n_bits

    prod_d = 1
    left = np.eye(1, 1, dtype=np.float64)
    for c in cores:
        l, p, r = c.shape
        left = (left @ c.reshape(l, p * r)).reshape(prod_d * p, r)
        prod_d *= p
    full = left.flatten()

    if full.size != N * N:
        raise ValueError(
            "Reconstruction size mismatch: got {}, expected {}.".format(full.size, N * N)
        )

    x = np.arange(N, dtype=np.int64)
    y = np.arange(N, dtype=np.int64)
    X, Y = np.meshgrid(x, y, indexing="xy")

    sig = np.zeros_like(X, dtype=np.int64)
    p_val = 1
    for k in range(n_bits):
        x_bit = (X >> k) & 1
        y_bit = (Y >> k) & 1
        sig += (2 * x_bit + y_bit) * p_val
        p_val *= 4

    return full[sig]


def load_data(
    path_1: str,
    path_2: Optional[str] = None,
    is_mps: bool = False,
    n_bits: int = 10,
):
    N = 2 ** n_bits
    expected_size = N * N

    if is_mps:
        if path_2 is None:
            raise ValueError("For is_mps=True, you must provide path_2 for v.")
        u = reconstruct_2d_field(load_mps_cores(path_1), n_bits)
        v = reconstruct_2d_field(load_mps_cores(path_2), n_bits)
        return u, v

    data = np.load(path_1, allow_pickle=True)
    if getattr(data, "ndim", None) == 1 and data.size == expected_size:
        data = data.reshape(N, N)

    if getattr(data, "ndim", None) == 3 and data.shape[0] == 4:
        return np.asarray(data[2]), np.asarray(data[3])

    if getattr(data, "ndim", None) == 3 and data.shape[0] == 2:
        return np.asarray(data[0]), np.asarray(data[1])

    if path_2 is not None:
        data_2 = np.load(path_2, allow_pickle=True)
        if getattr(data_2, "ndim", None) == 1 and data_2.size == expected_size:
            data_2 = data_2.reshape(N, N)
        return np.asarray(data), np.asarray(data_2)

    raise ValueError(
        "Unrecognized data format for {}: shape {}".format(
            path_1, getattr(data, "shape", None)
        )
    )


def find_closest_file(
    folder: str, prefix: str, target_time: float
) -> Tuple[Optional[str], Optional[float]]:
    if not os.path.exists(folder):
        return None, None

    files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".npy")]
    best_file = None
    best_time = None
    min_diff = float("inf")

    for f in files:
        stem = f[:-4]
        parts = stem.split("_")
        try:
            t = float(parts[-1])
        except ValueError:
            continue

        diff = abs(t - target_time)
        if diff < min_diff:
            min_diff = diff
            best_file = f
            best_time = t

    if best_file is None:
        return None, None

    return os.path.join(folder, best_file), best_time


def get_energy_spectrum(
    u: np.ndarray,
    v: np.ndarray,
    Lx: float = 1.0,
    Ly: float = 1.0,
    use_radian_k: bool = False,
    shell_average: bool = False,
):
    if u.shape != v.shape:
        raise ValueError("u and v must have same shape, got {} vs {}".format(u.shape, v.shape))

    ny, nx = u.shape
    dx = float(Lx) / float(nx)
    dy = float(Ly) / float(ny)

    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)

    Ntot = nx * ny
    E_hat = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) / (float(Ntot) ** 2)
    E_hat = np.fft.fftshift(E_hat)

    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    if use_radian_k:
        kx = 2.0 * np.pi * kx
        ky = 2.0 * np.pi * ky

    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K = np.sqrt(KX ** 2 + KY ** 2)

    k_max = int(np.floor(K.max()))
    if k_max < 1:
        return np.array([]), np.array([])

    k_bins = np.arange(0.5, k_max + 1.5, 1.0)
    k_1d = 0.5 * (k_bins[:-1] + k_bins[1:])
    E_1d = np.zeros_like(k_1d, dtype=np.float64)

    for i in range(len(k_1d)):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        if np.any(mask):
            E_1d[i] = E_hat[mask].mean() if shell_average else E_hat[mask].sum()

    return k_1d, E_1d


def main():
    parser = argparse.ArgumentParser(description="Plot 2D isotropic kinetic energy spectrum E(k).")
    parser.add_argument("--dns_dir", type=str, default="DNS_data")
    parser.add_argument("--rttd_dir", type=str, default="RTTD_data")
    parser.add_argument("--n", type=int, default=10, help="n_bits so N=2^n.")
    parser.add_argument("--t", type=float, default=2.0, help="Target time for spectrum.")
    parser.add_argument("--save", type=str, default="fig_tke.pdf")
    parser.add_argument("--Lx", type=float, default=1.0, help="Domain length in x (periodic).")
    parser.add_argument("--Ly", type=float, default=1.0, help="Domain length in y (periodic).")
    parser.add_argument("--radian_k", action="store_true", help="Use radian wavenumber (2π * cycles).")
    parser.add_argument("--shell_average", action="store_true", help="Shell-average instead of shell-sum.")
    args = parser.parse_args()

    t_target = args.t
    print("Finding files for t ≈ {} ...".format(t_target))

    dns_path, dns_t = find_closest_file(args.dns_dir, "velocity_", t_target)
    rttd_u_path, rttd_t = find_closest_file(args.rttd_dir, "u_time_", t_target)
    rttd_v_path, rttd_vt = find_closest_file(args.rttd_dir, "v_time_", t_target)

    if dns_path is None or rttd_u_path is None or rttd_v_path is None:
        raise FileNotFoundError(
            "Could not find required files.\n"
            "  DNS : folder={}, prefix='velocity_'\n"
            "  RTTD: folder={}, prefix='u_time_' and 'v_time_'".format(args.dns_dir, args.rttd_dir)
        )

    print("  DNS : {} (t={:.6g})".format(os.path.basename(dns_path), dns_t))
    print(
        "  RTTD: {} (t={:.6g}), {} (t={:.6g})".format(
            os.path.basename(rttd_u_path),
            rttd_t,
            os.path.basename(rttd_v_path),
            rttd_vt,
        )
    )

    u_d, v_d = load_data(dns_path, n_bits=args.n)
    u_r, v_r = load_data(rttd_u_path, rttd_v_path, is_mps=True, n_bits=args.n)

    k_dns, E_dns = get_energy_spectrum(
        u_d, v_d, Lx=args.Lx, Ly=args.Ly, use_radian_k=args.radian_k, shell_average=args.shell_average
    )
    k_rttd, E_rttd = get_energy_spectrum(
        u_r, v_r, Lx=args.Lx, Ly=args.Ly, use_radian_k=args.radian_k, shell_average=args.shell_average
    )

    if len(k_dns) == 0 or len(k_rttd) == 0:
        raise RuntimeError("Spectrum arrays are empty. Check inputs/domain parameters.")

    _, ax = plt.subplots(figsize=(8, 6))

    nmarks = 15
    max_idx = max(len(k_dns) - 1, 1)
    mark_idx = np.unique(
        np.clip(np.logspace(0, np.log10(max_idx), nmarks).astype(int), 0, len(k_dns) - 1)
    )

    ax.loglog(k_dns, E_dns, color="black", linestyle=":", label="DNS")
    ax.loglog(
        k_rttd,
        E_rttd,
        color="#d62728",
        linestyle="-",
        marker="s",
        markevery=mark_idx,
        markersize=7,
        markeredgewidth=2,
        label="RTTD",
    )

    ax.set_xlabel("k")
    ax.set_ylabel("Energy Spectrum")
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.save, dpi=300)
    print("Saved: {}".format(args.save))


if __name__ == "__main__":
    main()