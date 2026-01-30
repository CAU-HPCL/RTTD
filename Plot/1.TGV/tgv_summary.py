# tgv_summary.py
import os
import glob
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
        "figure.figsize": (18, 14),
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

CACHE_DIR = "processed_data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

CASES = [
    {"label": "RTTD", "chi": 32, "path": "pstt_run_10_32_1000_1_5/", "color": "#d62728", "marker": "s", "linestyle": "-"},
    {"label": "RTTD", "chi": 64, "path": "pstt_run_10_64_1000_1_5/", "color": "#8c564b", "marker": "^", "linestyle": "-"},
    {"label": "QIS",  "chi": 32, "path": "cuq_run_10_32_1000_1_5/",  "color": "#2ca02c", "marker": "o", "linestyle": "--"},
    {"label": "QIS",  "chi": 64, "path": "cuq_run_10_64_1000_1_5/",  "color": "#1f77b4", "marker": "v", "linestyle": "--"},
]

N_BITS = 10
N = 2**N_BITS
DX = 1.0 / N
NU = 1.0 / 1000.0

T_CUT = 0.3
OUT_PDF = "TGV.pdf"


def load_timings(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        loaded = np.load(file_path, allow_pickle=True)
        if isinstance(loaded, np.ndarray):
            return loaded.item()
        return loaded
    except (ValueError, pickle.UnpicklingError):
        try:
            with open(file_path, "r") as f:
                content = f.read().strip()
            if content.startswith("{"):
                return json.loads(content)
        except Exception:
            return None
    return None


def get_cumulative_wallclock(case):
    files = glob.glob(os.path.join(case["path"], "*comp_time*.npy"))
    if not files:
        return None, None

    timings = load_timings(files[0])
    if timings is None:
        return None, None

    keys = sorted(timings.keys(), key=lambda k: float(k))
    step_times = np.array([timings[k] for k in keys], dtype=np.float64)
    wall_clock_cum = np.cumsum(step_times)
    return step_times, wall_clock_cum


def stable_avg_step(step_times):
    if step_times is None or len(step_times) == 0:
        return None
    return float(np.mean(step_times[-200:])) if len(step_times) >= 200 else float(np.mean(step_times))


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


def analytical_energy(t):
    return 0.25 * np.exp(-16.0 * np.pi**2 * NU * t)


def analytical_field(t):
    x = np.linspace(0.0, 1.0 - DX, N)
    y = np.linspace(0.0, 1.0 - DX, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    decay = np.exp(-8.0 * np.pi**2 * NU * t)
    u = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * decay
    v = -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y) * decay
    return u, v


def process_case(case):
    safe_name = f"{case['label']}_{case['chi']}"
    cache_path = os.path.join(CACHE_DIR, f"{safe_name}.npz")

    step_times, wall_clock_full = get_cumulative_wallclock(case)

    if os.path.exists(cache_path):
        d = np.load(cache_path)
        res = {k: d[k] for k in d.files}
    else:
        root = case["path"]
        if not os.path.exists(root):
            return None

        u_files = sorted([f for f in os.listdir(root) if f.startswith("u_time_") and f.endswith(".npy")])
        times = []
        for f in u_files:
            try:
                times.append(float(f.split("_")[-1].replace(".npy", "")))
            except ValueError:
                pass
        times = sorted(times)

        t_list, l2_list, tke_list, div_list = [], [], [], []
        for t in times:
            u_f = os.path.join(root, f"u_time_{t if t > 0 else 0}.npy")
            v_f = os.path.join(root, f"v_time_{t if t > 0 else 0}.npy")
            if not (os.path.exists(u_f) and os.path.exists(v_f)):
                continue

            u = reconstruct_2d_field(load_mps_cores(u_f))
            v = reconstruct_2d_field(load_mps_cores(v_f))
            u_ref, v_ref = analytical_field(t)

            diff_norm = np.sqrt(np.sum((u - u_ref) ** 2 + (v - v_ref) ** 2))
            ref_norm = np.sqrt(np.sum(u_ref**2 + v_ref**2))
            l2 = diff_norm / ref_norm if ref_norm > 1e-15 else 0.0

            tke = 0.5 * DX**2 * np.sum(u**2 + v**2)

            du_dx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * DX)
            dv_dy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * DX)
            div = np.max(np.abs(du_dx + dv_dy))

            t_list.append(t)
            l2_list.append(l2)
            tke_list.append(tke)
            div_list.append(div)

        res = {"t": np.array(t_list), "l2": np.array(l2_list), "tke": np.array(tke_list), "div": np.array(div_list)}
        np.savez(cache_path, **res)

    res["avg_step_time"] = stable_avg_step(step_times)
    if wall_clock_full is not None and len(res["t"]) > 0:
        idx = np.linspace(0, len(wall_clock_full) - 1, len(res["t"])).astype(int)
        idx = np.clip(idx, 0, len(wall_clock_full) - 1)
        res["wall_clock_snaps"] = wall_clock_full[idx]
    else:
        res["wall_clock_snaps"] = np.zeros_like(res["t"])
    return res


def main():
    data_map = []
    for c in CASES:
        d = process_case(c)
        if d is not None and len(d["t"]) > 0:
            data_map.append((c, d))

    if not data_map:
        raise RuntimeError("No data found for any case.")

    t_max = min(min(np.max(d["t"]) for _, d in data_map), T_CUT)

    fig, axs = plt.subplots(2, 2)
    ax1, ax2 = axs[0, 0], axs[0, 1]
    ax3, ax4 = axs[1, 0], axs[1, 1]

    for case, d in data_map:
        mask = d["t"] <= t_max
        t = d["t"][mask]
        l2 = d["l2"][mask]
        tke = d["tke"][mask]
        div = d["div"][mask]

        wall = d["wall_clock_snaps"]
        tke_wall = d["tke"]

        lbl = rf"{case['label']}, $\chi={case['chi']}$"
        st = dict(
            color=case["color"],
            linestyle=case["linestyle"],
            marker=case["marker"],
            markevery=0.1,
            markersize=6,
            alpha=0.8,
            label=lbl,
        )

        ax1.plot(t, l2, **st)
        ax2.plot(t, tke, **st)
        ax3.plot(t, div, **st)
        ax4.plot(wall, tke_wall, **st)

    t_ref = np.linspace(0, t_max, 200)
    ax2.plot(t_ref, analytical_energy(t_ref), "k:", linewidth=2.5)

    ax1.set_title(r"(a) Relative $L_2$ Error")
    ax2.set_title(r"(b) Energy (Physics Time)")
    ax3.set_title(r"(c) Stability")
    ax4.set_title(r"(d) Energy (Wall-Clock Time)")

    ax1.set_xlabel("Physics Time")
    ax2.set_xlabel("Physics Time")
    ax3.set_xlabel("Physics Time")
    ax4.set_xlabel("Wall-Clock Time")

    ax1.set_ylabel("Relative Error")
    ax2.set_ylabel("Energy")
    ax3.set_ylabel("Stability")
    ax4.set_ylabel("Energy")

    ax1.set_yscale("log")
    ax3.set_yscale("log")

    ax1.set_xlim(0, t_max)
    ax2.set_xlim(0, t_max)
    ax3.set_xlim(0, t_max)

    phys_xticks = np.arange(0.0, t_max + 1e-12, 0.1)
    for ax in (ax1, ax2, ax3):
        ax.set_xticks(phys_xticks)

    handles, labels = ax4.get_legend_handles_labels()
    ana_line = Line2D([0], [0], color="k", linestyle=":", linewidth=2.5, label="Analytical")
    handles.insert(0, ana_line)
    labels.insert(0, "Analytical")

    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5)

    plt.tight_layout()
    plt.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()