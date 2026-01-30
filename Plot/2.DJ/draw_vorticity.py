import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.compression": 9,
    "font.size": 14,
})


def central_difference_1_8_x(f, dx):
    coeffs = np.array([4 / 5, -1 / 5, 4 / 105, -1 / 280], dtype=np.float64) / dx
    diff = np.zeros_like(f, dtype=np.float64)
    for i, coeff in enumerate(coeffs):
        diff += coeff * (np.roll(f, -i - 1, axis=1) - np.roll(f, i + 1, axis=1))
    return diff


def central_difference_1_8_y(f, dx):
    coeffs = np.array([4 / 5, -1 / 5, 4 / 105, -1 / 280], dtype=np.float64) / dx
    diff = np.zeros_like(f, dtype=np.float64)
    for i, coeff in enumerate(coeffs):
        diff += coeff * (np.roll(f, -i - 1, axis=0) - np.roll(f, i + 1, axis=0))
    return diff


def Dx(f, dx):
    return central_difference_1_8_x(f, dx)


def Dy(f, dx):
    return central_difference_1_8_y(f, dx)


def _load_any_numpy(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        if len(obj.files) == 0:
            raise ValueError(f"{path} is an empty .npz")
        obj = obj[obj.files[0]]
    return obj


def load_mps_cores(fname):
    data = _load_any_numpy(fname)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()
    cores = data if isinstance(data, list) else list(data)
    return [np.asarray(i, dtype=np.float64) for i in cores]


def reconstruct_2d_field(cores, n_bits):
    N = 2 ** n_bits

    prod_d = 1
    left = np.eye(1, 1, dtype=np.float64)
    for c in cores:
        c = np.asarray(c, dtype=np.float64)
        l, p, r = c.shape
        left = (left @ c.reshape(l, p * r)).reshape(prod_d * p, r)
        prod_d *= p
    full = left.flatten()

    if full.size != N * N:
        raise ValueError(f"MPS reconstruction size mismatch: got {full.size}, expected {N * N}")

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


def load_data(path_1, path_2=None, is_mps=False, n_bits=10):
    if is_mps:
        if not path_2:
            raise ValueError("MPS mode requires both --path (u) and --v_path (v).")
        print(f"Loading MPS cores (n={n_bits})...")
        u = reconstruct_2d_field(load_mps_cores(path_1), n_bits)
        v = reconstruct_2d_field(load_mps_cores(path_2), n_bits)
        return u, v

    print(f"Loading: {path_1}")
    data = _load_any_numpy(path_1)

    if isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[0] == 4:
        print("Detected Simulation Format (4, N, N). Extracting u, v from indices 2, 3.")
        return data[2], data[3]

    if isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[0] == 2:
        print("Detected Stacked Format (2, N, N).")
        return data[0], data[1]

    if path_2:
        print(f"Loading second file: {path_2}")
        data_2 = _load_any_numpy(path_2)

        N = 2 ** n_bits
        if isinstance(data, np.ndarray) and data.ndim == 1 and data.size == N * N:
            data = data.reshape(N, N)
        if isinstance(data_2, np.ndarray) and data_2.ndim == 1 and data_2.size == N * N:
            data_2 = data_2.reshape(N, N)

        if isinstance(data, np.ndarray) and data.shape == (N, N):
            print("Note: transposing (u,v) from separate files to match expected layout.")
            return data.T, data_2.T
        return data, data_2

    raise ValueError(
        f"Unrecognized shape {getattr(data, 'shape', None)}. Is this a (4, N, N) simulation file?"
    )


def main():
    parser = argparse.ArgumentParser(description="Plot Vorticity from Simulation Output")
    parser.add_argument("--path", "-u", type=str, required=True, help="Path to .npy file")
    parser.add_argument("--v_path", "-v", type=str, default=None, help="Path to v file (optional)")
    parser.add_argument("--mps", action="store_true", help="Use MPS reconstruction")
    parser.add_argument("--n", type=int, default=10, help="Grid size 2^n")
    parser.add_argument("--save", type=str, default=None, help="Save to file")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed max vorticity for consistent colorbar across plots")
    args = parser.parse_args()

    if "velocity_" in args.path:
        match = re.search(r"velocity_(\d+)_", args.path)
        if match:
            detected_n = int(match.group(1))
            if detected_n != args.n:
                print(f"Auto-detected n={detected_n} from filename (overriding default {args.n})")
                args.n = detected_n

    try:
        u, v = load_data(args.path, args.v_path, args.mps, args.n)
    except Exception as e:
        print(f"Error: {e}")
        return

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    N = 2 ** args.n
    dx = 1 / (N - 1)

    x = np.linspace(0.0, 1.0, N, endpoint=True)
    y = np.linspace(0.0, 1.0, N, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing="xy")

    print("Calculating Vorticity...")
    vorticity = Dx(v, dx) - Dy(u, dx)

    n_s = max(1, 2 ** (args.n - 4))
    plt.figure()

    if args.vmax is not None:
        w_max = float(args.vmax)
        print(f"Using fixed color scale: vmax={w_max}")
    else:
        w_max = np.nanmax(np.abs(vorticity))
        if (not np.isfinite(w_max)) or w_max == 0:
            w_max = 1e-5
        print(f"Auto-scaled vmax={w_max} (Warning: inconsistent across plots if not fixed)")

    levels = np.linspace(-w_max, w_max, 100)

    cf = plt.contourf(
        X, Y, vorticity,
        levels=levels,
        cmap="seismic",
        vmin=-w_max,
        vmax=w_max,
        extend="both",
    )

    for c in cf.collections:
        c.set_rasterized(True)

    plt.quiver(
        X[::n_s, ::n_s],
        Y[::n_s, ::n_s],
        u[::n_s, ::n_s],
        v[::n_s, ::n_s],
        color="black",
        rasterized=True,
    )

    plt.xlim((0, 1))
    plt.ylim((0, 1))

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches="tight", pad_inches=0)
        print(f"Saved to {args.save}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()