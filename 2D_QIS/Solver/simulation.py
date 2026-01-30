import os
from QIS_2D import QI_CFD

METHODS = ["rttd", "cuq"]
CHIS = [16, 32, 64]
N_BITS_LIST = [10, 11, 12]

Re = 100000.0
T_FINAL = 2.0
L = 1
MU = 2.5e5
SOLVER = "cg"
SAVE_NUMBER = 100


def main():
    for n_bits in N_BITS_LIST:
        for method in METHODS:
            for chi in CHIS:
                out_dir = f"{method}/results/run_{n_bits}_{chi}_{Re}"
                os.makedirs(out_dir, exist_ok=True)

                sim = QI_CFD(method=method)
                sim.init_params(n_bits, L, chi, chi, T_FINAL, MU, Re, out_dir, SOLVER, SAVE_NUMBER)

                sim.meas_comp_time = True
                sim.comp_time_path = f"{out_dir}/{method}_comp_time_run_{n_bits}_{chi}_{Re}.npy"

                sim.build_initial_fields() # for DJ
                # sim.build_initial_fields_taylor() # for TGV
                sim.time_evolution()


if __name__ == "__main__":
    main()