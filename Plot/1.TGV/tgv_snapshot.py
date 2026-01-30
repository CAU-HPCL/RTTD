# tgv_snapshot.py
import os
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.size": 20,
        "font.family": "serif",
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "lines.linewidth": 2.0,
        "figure.figsize": (24, 6),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 9,
        "path.simplify": True,
        "path.simplify_threshold": 0.8,
        "mathtext.fontset": "cm",
    }
)

CASE_PATH = "pstt_run_10_32_1000_1_5/"
TIME_TO_PLOT = 0.3
OUT_PDF = "TGV_err.pdf"

N_BITS = 10
N = 2**N_BITS
DX = 1.0 / N
NU = 1.0 / 1000.0


def load_mps_cores(fname: str):
    data = np.load(fname, allow_pickle=True)
    if isinstance(data, list):
        cores = data
    else:
        cores = list(data)
    return [np.asarray(c, dtype=np.float64) for c in cores]


def reconstruct_2d_field(cores):
    prod_d = 1
    left = np.eye(1, 1)

    for c in cores:
        l, p, r = c.shape
        left = (left @ c.reshape(l, p * r)).reshape(prod_d * p, r)
        prod_d *= p

    full = left.flatten()
    field = np.zeros((N, N), dtype=np.float64)

    for ix in range(N):
        for iy in range(N):
            sig = 0
            p_val = 1
            for k in range(N_BITS):
                sig += ((ix >> k) & 1) * 2 * p_val + ((iy >> k) & 1) * p_val
                p_val *= 4
            field[ix, iy] = full[sig]

    return field


def periodic_gradient(f, dx, axis: int):
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2 * dx)


def analytical_field(t: float):
    x = np.linspace(0.0, 1.0 - DX, N)
    y = np.linspace(0.0, 1.0 - DX, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    decay = np.exp(-8.0 * np.pi**2 * NU * t)
    u = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * decay
    v = -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y) * decay
    return u, v


def find_closest_time(case_path: str, target_t: float) -> float:
    files = [f for f in os.listdir(case_path) if f.startswith("u_time_") and f.endswith(".npy")]
    if not files:
        raise FileNotFoundError(f"No u_time_*.npy in {case_path}")

    times = []
    for f in files:
        try:
            t = float(f.split("_")[-1].replace(".npy", ""))
            times.append(t)
        except ValueError:
            pass

    if not times:
        raise RuntimeError(f"Could not parse times from u_time_*.npy in {case_path}")

    return min(times, key=lambda x: abs(x - target_t))


def main():
    t = find_closest_time(CASE_PATH, TIME_TO_PLOT)
    u_f = os.path.join(CASE_PATH, f"u_time_{t if t > 0 else 0}.npy")
    v_f = os.path.join(CASE_PATH, f"v_time_{t if t > 0 else 0}.npy")

    if not (os.path.exists(u_f) and os.path.exists(v_f)):
        raise FileNotFoundError(f"Missing snapshot files:\n  {u_f}\n  {v_f}")

    u_num = reconstruct_2d_field(load_mps_cores(u_f))
    v_num = reconstruct_2d_field(load_mps_cores(v_f))
    u_ref, v_ref = analytical_field(t)

    vort_num = periodic_gradient(v_num, DX, axis=0) - periodic_gradient(u_num, DX, axis=1)
    vort_ref = periodic_gradient(v_ref, DX, axis=0) - periodic_gradient(u_ref, DX, axis=1)

    vel_err = np.sqrt((u_num - u_ref) ** 2 + (v_num - v_ref) ** 2)
    vort_err = np.abs(vort_num - vort_ref)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    w_lim = np.max(np.abs(vort_num))
    if w_lim == 0:
        w_lim = 1.0

    im1 = axes[0].imshow(
        vort_num.T, extent=[0, 1, 0, 1], origin="lower", cmap="seismic", vmin=-w_lim, vmax=w_lim
    )
    axes[0].set_title(r"Vorticity $\omega$")

    im2 = axes[1].imshow(vel_err.T, extent=[0, 1, 0, 1], origin="lower", cmap="inferno", vmin=0)
    axes[1].set_title(r"Velocity Error $|u-u_{\mathrm{ana}}|$")

    im3 = axes[2].imshow(vort_err.T, extent=[0, 1, 0, 1], origin="lower", cmap="inferno", vmin=0)
    axes[2].set_title(r"Vorticity Error $|\omega-\omega_{\mathrm{ana}}|$")

    for ax, im in zip(axes, [im1, im2, im3]):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    print(f"Saved: {OUT_PDF} (t={t:.6f})")


if __name__ == "__main__":
    main()