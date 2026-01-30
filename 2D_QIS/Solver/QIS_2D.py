# Copyright 2024 Leonhard Hoelscher. All Rights Reserved.
# Modifications Copyright 2026 Kiet Tuan Pham.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from differential_mpo import *
from differential_operators_numpy import *

import cupy as cp
from cuquantum import cutensornet as cutn
from cuquantum.cutensornet.tensor import decompose, SVDMethod
from cuquantum.cutensornet.experimental import contract_decompose, make_network, rttd
from cuquantum import contract, tensor, OptimizerOptions, Network
import sys
import json
import time

import subprocess as sp
import importlib.util
import threading


class InternalProfiler:
    """Monitors CuPy VRAM usage in a background thread."""
    def __init__(self, mempool, poll_interval_ms=1):
        self.mempool = mempool
        self.poll_interval_s = poll_interval_ms / 1000.0
        self.peak_vram_bytes = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor, daemon=True)

    def _monitor(self):
        try:
            while not self._stop_event.is_set():
                current_used = self.mempool.used_bytes()
                if current_used > self.peak_vram_bytes:
                    self.peak_vram_bytes = current_used
                time.sleep(self.poll_interval_s)
        except Exception:
            pass

    def start(self):
        self.peak_vram_bytes = self.mempool.used_bytes()
        self._stop_event.clear()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=0.1)

    @property
    def peak_vram_mib(self):
        return self.peak_vram_bytes / (1024 * 1024)


def check_tensor_for_nan(tensor, name, context=""):
    if tensor is None:
        print(f"❌ None tensor in {name} ({context}) - cannot fix")
        return None
    if tensor.size == 0:
        print(f"⚠️ Empty tensor in {name} ({context}) - skipping check")
        return tensor

    has_nan = cp.any(cp.isnan(tensor))
    has_inf = cp.any(cp.isinf(tensor))
    t_norm = float(cp.linalg.norm(tensor))

    if has_nan or has_inf:
        print(f"❌ NaN/Inf in {name} ({context}): norm={t_norm:.2e}, shape={tensor.shape}")
        print(f"   Min/max: {float(cp.min(tensor)):.2e}/{float(cp.max(tensor)):.2e}")
        tensor = cp.nan_to_num(tensor, nan=0.0, posinf=1e5, neginf=-1e5)
        if t_norm > 1e-10:
            tensor /= t_norm
            t_norm = 1.0
        else:
            total_elements = np.prod(tensor.shape)
            if total_elements > 0:
                tensor.fill(1.0 / cp.sqrt(float(total_elements)))
                t_norm = 1.0
            else:
                print(f"⚠️ Still empty after fix in {name} - set to zero")
                tensor.fill(0.0)
        print(f"   Fixed: now norm={cp.linalg.norm(tensor):.2e}")
        return tensor

    if t_norm > 1e10 or t_norm < 1e-12:
        print(f"⚠️ Extreme norm {t_norm:.2e} in {name} ({context}) - rescaling")
        if t_norm > 0:
            tensor /= t_norm
        print(f"   Rescaled to 1.0")
        return tensor

    return tensor


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def _sync():
    cp.cuda.Stream.null.synchronize()


class QI_CFD:
    def __init__(self, method):
        self.handle = cutn.create()
        self.options = {'handle': self.handle}
        self.networks = {}
        self.method = method

        self.n_bits = 10
        self.N = 2**self.n_bits
        self.L = 1
        self.chi = 33
        self.chi_mpo = 33
        self.dt = 0.1*2.0**-(self.n_bits-1)
        self.T = 2
        self.dx = 1 / (2**self.n_bits - 1)
        self.mu = 250000.0
        self.Re = 200*1e3
        self.solver = "cg"
        self.path = None
        self.save_number = 100
        self.meas_comp_time = False
        self.comp_time_path = None
        self.max_sweeps = 20
        self.t = 0

        self.U_init = None
        self.V_init = None

    def init_params(self, n_bits, L, chi, chi_mpo, T, mu, Re, path, solver, save_number=100):
        self.n_bits = n_bits
        self.N = 2**self.n_bits
        self.L = L
        self.chi = chi
        self.chi_mpo = chi_mpo
        self.dt = 0.1*2.0**-(self.n_bits-1)
        self.T = T
        self.dx = 1 / (2**self.n_bits - 1)
        self.mu = mu
        self.Re = Re
        self.path = path
        self.solver = solver
        self.save_number = save_number
        print("Initialized parameters")

    def copy_tn(self, tensor_list):
        return [tensor.copy() for tensor in tensor_list]

    def multiply_mps_mpo(self, mps, mpo, algorithm, options=None):
        t = contract('ipj,kplm->ijlm', mps[0], mpo[0], options=options)
        output_mps = []
        for i in range(1, self.n_bits):
            mps_node, _, t = contract_decompose(
                'ijlm,jqr,lqsn->imx,xrsn', t, mps[i], mpo[i],
                algorithm=algorithm, options=options
            )
            output_mps.append(mps_node)
        t = t.reshape(-1, 4, 1)
        output_mps.append(t)
        return output_mps

    def multiply_mpo_mpo(self, mpo_2, mpo_1, algorithm, options=None):
        t = contract('akbp,lprP->akbrP', mpo_1[0], mpo_2[0], options=options)
        output_mpo = []
        for i in range(1, self.n_bits):
            if self.method == 'cuq':
                mpo, _, t = contract_decompose(
                    'akbrP,bKcD,rDeF->akxP,xKceF', t, mpo_1[i], mpo_2[i],
                    algorithm=algorithm, options=options
                )
            elif self.method == 'rttd':
                mpo, _, t = rttd(
                    'akbrP,bKcD,rDeF->akxP,xKceF', t, mpo_1[i], mpo_2[i],
                    algorithm=algorithm, options=options
                )
            elif self.method == 'cuq_rsvd':
                random_algorithm = {
                    "qr_method": False,
                    "svd_method": {
                        "partition": "V",
                        "max_extent": 32,
                        "algorithm": "gesvd",
                    },
                }
                mpo, _, t = contract_decompose(
                    'akbrP,bKcD,rDeF->akxP,xKceF', t, mpo_1[i], mpo_2[i],
                    algorithm=random_algorithm, options=options
                )
            elif self.method == 'torch':
                mpo, _, t = rttd(
                    'akbrP,bKcD,rDeF->akxP,xKceF', t, mpo_1[i], mpo_2[i],
                    algorithm=algorithm, options=options
                )
            output_mpo.append(mpo)

        t = t.reshape(-1, 4, 1, 4)
        output_mpo.append(t)
        return output_mpo

    def canonicalize_mps_tensors(self, a, b, absorb='right', options=None):
        if absorb == 'right':
            a, r = tensor.decompose('ipj->ipx,xj', a, options=options)
            b = contract('xj,jpk->xpk', r, b, options=options)
        elif absorb == 'left':
            b, r = tensor.decompose('jpk->xpk,jx', b, options=options)
            a = contract('jx,ipj->ipx', r, a, options=options)
        else:
            raise ValueError("absorb must be either left or right")
        return a, b

    def right_canonicalize_mps(self, mps_tensors, start, end, options=None):
        assert end >= start
        for i in range(start, end):
            mps_tensors[i:i+2] = self.canonicalize_mps_tensors(
                *mps_tensors[i:i+2], absorb='right', options=options
            )
        return mps_tensors

    def left_canonicalize_mps(self, mps_tensors, start, end, options=None):
        assert start >= end
        for i in range(start, end, -1):
            mps_tensors[i-1:i+1] = self.canonicalize_mps_tensors(
                *mps_tensors[i-1:i+1], absorb='left', options=options
            )
        return mps_tensors

    def canonical_center(self, mps, center, options=None):
        mps_r = self.right_canonicalize_mps(mps, 0, center, options=options)
        mps_rl = self.left_canonicalize_mps(mps_r, self.n_bits-1, center, options=options)
        return mps_rl

    def shift_canonical_center(self, mps, center, initial=None, options=None):
        if initial is None:
            return self.canonical_center(mps, center, options)
        elif initial > center:
            for i in range(initial, center, -1):
                mps[i-1:i+1] = self.canonicalize_mps_tensors(
                    *mps[i-1:i+1], absorb='left', options=options
                )
            return mps
        else:
            for i in range(initial, center):
                mps[i:i+2] = self.canonicalize_mps_tensors(
                    *mps[i:i+2], absorb='right', options=options
                )
            return mps

    def J(self, X, Y, u_0, y_min=0.4, y_max=0.6, h=0.005):
        return u_0/2*(np.tanh((Y-y_min)/h)-np.tanh((Y-y_max)/h)-1), np.zeros_like(Y)

    def d_1(self, X, Y, y_min=0.4, y_max=0.6, h=0.005, L_box=1):
        return 2*L_box/h**2*((Y-y_max)*np.exp(-(Y-y_max)**2/h**2)+(Y-y_min)*np.exp(-(Y-y_min)**2/h**2))*(np.sin(8*np.pi*X/L_box)+np.sin(24*np.pi*X/L_box)+np.sin(6*np.pi*X/L_box))

    def d_2(self, X, Y, y_min=0.4, y_max=0.6, h=0.005, L_box=1):
        return np.pi*(np.exp(-(Y-y_max)**2/h**2)+np.exp(-(Y-y_min)**2/h**2))*(8*np.cos(8*np.pi*X/L_box)+24*np.cos(24*np.pi*X/L_box)+6*np.cos(6*np.pi*X/L_box))

    def D(self, X, Y, u_0, y_min, y_max, h, L_box):
        d1 = self.d_1(X, Y, y_min, y_max, h, L_box)
        d2 = self.d_2(X, Y, y_min, y_max, h, L_box)
        delta = u_0/(40*np.max(np.sqrt(d1**2+d2**2)))
        return delta*d1, delta*d2

    def initial_fields(self, y_min, y_max, h, u_max):
        x = np.linspace(0, self.L-self.dx, self.N)
        y = np.linspace(0, self.L-self.dx, self.N)
        Y, X = np.meshgrid(y, x)
        U, V = self.J(X, Y, u_max, y_min, y_max, h)
        dU, dV = self.D(X, Y, u_max, y_min, y_max, h, self.L)
        U = U + dU
        V = V + dV
        return U, V

    def get_A_index(self, binary):
        return int(binary[::2] + binary[1::2], 2)

    def svd(self, mat, chi):
        U, S, V = decompose('ij->ik,kj', mat, method=SVDMethod(max_extent=chi))
        return U, np.diag(S), V

    def compress_mps(self, input_mps, chi, curr_center=None, options=None):
        mps = [tensor.copy() for tensor in input_mps]
        mps = self.shift_canonical_center(mps, 0, curr_center)
        n = len(mps)
        t = mps[0]

        if self.method == 'cuq':
            mult_algorithm = {'qr_method': False, 'svd_method': {'partition': 'V', 'max_extent': chi}}
            for i in range(1, n):
                mps[i-1], _, t = contract_decompose('lpr,rPR->lpx,xPR', t, mps[i], algorithm=mult_algorithm, options=options)
            mps[-1] = t

            mult_algorithm = {'qr_method': False, 'svd_method': {'partition': 'U', 'max_extent': chi}}
            for i in range(n-2, -1, -1):
                t, _, mps[i+1] = contract_decompose('lpr,rPR->lpx,xPR', mps[i], t, algorithm=mult_algorithm, options=options)
            mps[0] = t

        elif self.method == 'cuq_qr':
            start = time.time()

            mult_algorithm = {
                'qr_method': False,
                'svd_method': {'partition': 'V', 'max_extent': chi, 'use_rttd': True}
            }
            for i in range(1, n):
                mps[i-1], _, t = make_network('lpr,rPR->lpx,xPR', t, mps[i], algorithm=mult_algorithm, options=options)
            mps[-1] = t

            mult_algorithm = {
                'qr_method': False,
                'svd_method': {'partition': 'U', 'max_extent': chi, 'use_rttd': True}
            }
            for i in range(n-2, -1, -1):
                t, _, mps[i+1] = make_network('lpr,rPR->lpx,xPR', mps[i], t, algorithm=mult_algorithm, options=options)
            mps[0] = t

            end = time.time()
            print(f"Compression time (cuq_qr): {end-start} seconds")

        elif self.method == 'rttd' or self.method == 'torch':
            mult_algorithm = {'qr_method': False, 'svd_method': {'partition': 'V', 'max_extent': chi}}
            for i in range(1, n):
                mps[i-1], _, t = contract_decompose('lpr,rPR->lpx,xPR', t, mps[i], algorithm=mult_algorithm, options=options)
            mps[-1] = t

            mult_algorithm = {'qr_method': False, 'svd_method': {'partition': 'U', 'max_extent': chi}}
            for i in range(n-2, -1, -1):
                t, _, mps[i+1] = contract_decompose('lpr,rPR->lpx,xPR', mps[i], t, algorithm=mult_algorithm, options=options)
            mps[0] = t

        return mps

    def convert_to_MPS2D(self, A, chi=None):
        A_vec = A.reshape((1, -1))
        w = '0' * 2 * self.n_bits
        B_vec = np.zeros(4**self.n_bits).reshape((1, -1))

        for _ in range(4**self.n_bits):
            A_index = self.get_A_index(w)
            B_index = int(w, 2)
            w = bin(B_index + 1)[2:].zfill(2*self.n_bits)
            B_vec[0, B_index] = A_vec[0, A_index]

        node = B_vec
        MPS = []

        for _ in range(self.n_bits - 1):
            m, n = node.shape
            node = node.reshape((4*m, int(n/4)))
            U, S, V = self.svd(node, chi)
            MPS.append(U)
            node = np.matmul(S, V)

        m, n = node.shape
        node = node.reshape((4*m, int(n/4)))
        MPS.append(node)
        return MPS

    def convert_to_VF2D(self, MPS):
        node_L = MPS[0]
        for i in range(1, self.n_bits):
            m, n = node_L.shape
            node_R = MPS[i].reshape((n, -1))
            node_L = np.matmul(node_L, node_R)
            m, n = node_L.shape
            node_L = node_L.reshape((4*m, int(n/4)))
        B_vec = node_L.reshape((1, -1))

        w = '0' * 2 * self.n_bits
        A_vec = np.zeros(4**self.n_bits).reshape((1, -1))

        for _ in range(4**self.n_bits):
            A_index = self.get_A_index(w)
            B_index = int(w, 2)
            w = bin(B_index + 1)[2:].zfill(2*self.n_bits)
            A_vec[0, A_index] = B_vec[0, B_index]

        return A_vec.reshape((self.N, self.N))

    def convert_MPS_to_cupy(self, tensor_list, dim_p):
        arrays = []
        for tensor in tensor_list:
            m, n = tensor.shape
            dim_left_bond = int(m/dim_p)
            dim_right_bond = n
            data = tensor.reshape((dim_left_bond, dim_p, dim_right_bond))
            arrays.append(cp.asarray(data))
        return arrays

    def convert_cupy_to_MPS(self, mps):
        arrays = []
        for tensor in mps:
            l, p, r = tensor.shape
            arrays.append(tensor.reshape((l*p, r)))
        return arrays

    def convert_ls(self, A):
        A_vec = A.reshape((1, -1))
        w = '0' * 2 * self.n_bits
        B_vec = np.zeros(4**self.n_bits).reshape((1, -1))

        for _ in range(4**self.n_bits):
            A_index = self.get_A_index(w)
            B_index = int(w, 2)
            w = bin(B_index + 1)[2:].zfill(2*self.n_bits)
            B_vec[0, B_index] = A_vec[0, A_index]

        return B_vec.reshape((self.N, self.N))

    def convert_back(self, A):
        A_vec = A.reshape((1, -1))
        w = '0' * 2 * self.n_bits
        B_vec = np.zeros(4**self.n_bits).reshape((1, -1))

        for _ in range(4**self.n_bits):
            A_index = self.get_A_index(w)
            B_index = int(w, 2)
            w = bin(B_index + 1)[2:].zfill(2*self.n_bits)
            B_vec[0, A_index] = A_vec[0, B_index]

        return B_vec.reshape((self.N, self.N))


    def hadamard_product_MPO(self, mps, options=None):
        # prepares as MPO from an MPS to perform a hadamard product with another MPS
        # mps:      o--o--o--o--o
        #           |  |  |  |  |
        #
        #           |  |  |  |  |
        # k_delta:  k  k  k  k  k
        #          /\ /\ /\ /\ /\
        #
        # -> mpo:   o--o--o--o--o
        #          /\ /\ /\ /\ /\
        
        k_delta = cp.zeros((4, 4, 4), dtype='float64')  # initialize kronecker delta as np.array
        for i in range(4):
            k_delta[i, i, i] = 1    # only set variables to one where each index is the same
        mpo = self.copy_tn(mps)
        for i, tensor in enumerate(mpo):
            mpo[i] = contract('ijk, jlm->ilkm', tensor, k_delta, options=options)
        
        return mpo   # return the MPO


    def get_precontracted_LR_mps_mps(self, mps_2, mps_1, center=0, options=None):
        # prepare precontracted networks for dmrg sweeps
        # mps_1:    o--o-- center --o--o
        #           |  |            |  |
        # mps_2:    o--o-- center --o--o
        #           left            right networks

        left_networks = [None]*self.n_bits    # create a list containing the contracted left network for each site
        right_networks = [None]*self.n_bits   # create a list containing the contracted right network for each site

        # handle boundary networks
        dummy_t = cp.ones((1, 1))   # create a dummy network consisting of a 1
        left_networks[0] = dummy_t
        right_networks[-1] = dummy_t

        o = OptimizerOptions(path=[(0, 2), (0, 1)])

        if 'g_p_L_mps_mps' not in self.networks.keys():
            self.networks['g_p_L_mps_mps'] = [None]*self.n_bits
            self.networks['g_p_R_mps_mps'] = [None]*self.n_bits

        # from left to right
        for i in range(center):
            A = mps_1[i]
            B = mps_2[i]
            F = left_networks[i]

            if self.networks['g_p_L_mps_mps'][i] is None:
                self.networks['g_p_L_mps_mps'][i] = Network('apb, cpd, ac->bd', A, B, F, options=options)
                path, info = self.networks['g_p_L_mps_mps'][i].contract_path(optimize=o)
            else:
                self.networks['g_p_L_mps_mps'][i].reset_operands(A, B, F)

            F_new = self.networks['g_p_L_mps_mps'][i].contract(release_workspace=True)
            # self.networks['g_p_L_mps_mps'][i].workspace_ptr = None
            # F_new = contract('apb, cpd, ac->bd', A, B, F, options=options, optimize=o)
            
            left_networks[i+1] = F_new
        
        # from right to left
        for i in range(self.n_bits-1, center, -1):
            A = mps_1[i]
            B = mps_2[i]
            F = right_networks[i]

            if self.networks['g_p_R_mps_mps'][i] is None:
                self.networks['g_p_R_mps_mps'][i] = Network('apb, cpd, bd->ac', A, B, F, options=options)
                path, info = self.networks['g_p_R_mps_mps'][i].contract_path(optimize=o)
            else:
                self.networks['g_p_R_mps_mps'][i].reset_operands(A, B, F)

            F_new = self.networks['g_p_R_mps_mps'][i].contract(release_workspace=True)
            # self.networks['g_p_R_mps_mps'][i].workspace_ptr = None
            # F_new = contract('apb, cpd, bd->ac', A, B, F, options=options, optimize=o)
            
            right_networks[i-1] = F_new

        return left_networks, right_networks


    def get_precontracted_LR_mps_mpo(self, mps_2, mpo, mps_1, center=0, extra='', options=None):
        # prepare precontracted networks for dmrg sweeps
        # mps_1:    o--o-- center --o--o
        #           |  |            |  |
        # mpo:      0--0--        --0--0
        #           |  |            |  |
        # mps_2:    o--o-- center --o--o
        #           left            right networks

        left_networks = [None]*self.n_bits    # create a list containing the contracted left network for each site
        right_networks = [None]*self.n_bits   # create a list containing the contracted right network for each site

        # handle boundary networks
        dummy_t = cp.ones((1, 1, 1))   # create a dummy network consisting of a 1
        left_networks[0] = dummy_t
        right_networks[-1] = dummy_t

        o = OptimizerOptions(path=[(0, 3), (0, 2), (0, 1)])
        
        if f'g_p_L_mps_mpo{extra}' not in self.networks.keys():
            self.networks[f'g_p_L_mps_mpo{extra}'] = [None]*self.n_bits
            self.networks[f'g_p_R_mps_mpo{extra}'] = [None]*self.n_bits

        # from left to right
        for i in range(center):
            A = mps_1[i]
            B = mps_2[i]
            W = mpo[i]
            F = left_networks[i]

            if self.networks[f'g_p_L_mps_mpo{extra}'][i] is None:
                self.networks[f'g_p_L_mps_mpo{extra}'][i] = Network('apb, lprP, cPd, alc->brd', A, W, B, F, options=options)
                path, info = self.networks[f'g_p_L_mps_mpo{extra}'][i].contract_path(optimize=o)
            else:
                self.networks[f'g_p_L_mps_mpo{extra}'][i].reset_operands(A, W, B, F)

            F_new = self.networks[f'g_p_L_mps_mpo{extra}'][i].contract(release_workspace=True)
            # self.networks[f'g_p_L_mps_mpo{extra}'][i].workspace_ptr = None
            # F_new = contract('apb, lprP, cPd, alc->brd', A, W, B, F, options=options, optimize=o)
            
            left_networks[i+1] = F_new

        # from right to left
        for i in range(self.n_bits-1, center, -1):
            A = mps_1[i]
            B = mps_2[i]
            W = mpo[i]
            F = right_networks[i]

            if self.networks[f'g_p_R_mps_mpo{extra}'][i] is None:
                self.networks[f'g_p_R_mps_mpo{extra}'][i] = Network('apb, lprP, cPd, brd->alc', A, W, B, F, options=options)
                path, info = self.networks[f'g_p_R_mps_mpo{extra}'][i].contract_path(optimize=o)
            else:
                self.networks[f'g_p_R_mps_mpo{extra}'][i].reset_operands(A, W, B, F)

            F_new = self.networks[f'g_p_R_mps_mpo{extra}'][i].contract(release_workspace=True)
            # self.networks[f'g_p_R_mps_mpo{extra}'][i].workspace_ptr = None
            # F_new = contract('apb, lprP, cPd, brd->alc', A, W, B, F, options=options, optimize=o)
            
            right_networks[i-1] = F_new

        return left_networks, right_networks


    def update_precontracted_LR_mps_mps(self, F, B, A, LR, pos, extra='', options=None):
        # update the precontracted networks for dmrg sweeps
        #                        F--A--
        # For LR='L' contract :  F  |
        #                        F--B--
        #
        #                        --A--F
        # For LR='R' contract :    |  F
        #                        --B--F

        operands = [A, B, F]
        o = OptimizerOptions(path=[(0, 2), (0, 1)])

        if f'u_p_{LR}_mps_mps{extra}' not in self.networks.keys():
            self.networks[f'u_p_{LR}_mps_mps{extra}'] = [None]*self.n_bits

        if self.networks[f'u_p_{LR}_mps_mps{extra}'][pos] is None:
            if LR == 'L':
                self.networks[f'u_p_L_mps_mps{extra}'][pos] = Network('apb, cpd, ac->bd', *operands, options=options)
            elif LR == 'R':
                self.networks[f'u_p_R_mps_mps{extra}'][pos] = Network('apb, cpd, bd->ac', *operands, options=options)
            path, info = self.networks[f'u_p_{LR}_mps_mps{extra}'][pos].contract_path(optimize = o)
        else:
            self.networks[f'u_p_{LR}_mps_mps{extra}'][pos].reset_operands(*operands)
        
        F_new = self.networks[f'u_p_{LR}_mps_mps{extra}'][pos].contract(release_workspace=True)
        # self.networks[f'u_p_{LR}_mps_mps{extra}'][pos].workspace_ptr = None
        
        return F_new


    def update_precontracted_LR_mps_mpo(self, F, B, W, A, LR, pos, extra='', options=None):
        # update the precontracted networks for dmrg sweeps
        #                        F--A--
        #                        F  |
        # For LR='L' contract :  F--W--
        #                        F  |
        #                        F--B--
        #
        #                        --A--F
        #                          |  F
        # For LR='R' contract :  --W--F
        #                          |  F
        #                        --B--F

        operands = [A, W, B, F]
        o = OptimizerOptions(path=[(0, 3), (0, 2), (0, 1)])

        if f'u_p_{LR}_mps_mpo{extra}' not in self.networks.keys():
            self.networks[f'u_p_{LR}_mps_mpo{extra}'] = [None]*self.n_bits

        if self.networks[f'u_p_{LR}_mps_mpo{extra}'][pos] is None:
            if LR == 'L':
                self.networks[f'u_p_L_mps_mpo{extra}'][pos] = Network('apb, lprP, cPd, alc->brd', *operands, options=options)
            elif LR == 'R':
                self.networks[f'u_p_R_mps_mpo{extra}'][pos] = Network('apb, lprP, cPd, brd->alc', *operands, options=options)
            path, info = self.networks[f'u_p_{LR}_mps_mpo{extra}'][pos].contract_path(optimize = o)
        else:
            self.networks[f'u_p_{LR}_mps_mpo{extra}'][pos].reset_operands(*operands)
        
        F_new = self.networks[f'u_p_{LR}_mps_mpo{extra}'][pos].contract(release_workspace=True)
        # self.networks[f'u_p_{LR}_mps_mpo{extra}'][pos].workspace_ptr = None
        
        return F_new


    def Ax(self, H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y, pos, options=None):
        # gives tensors corresponding to A*x
        # A = (1 - H_11, -H_12)
        #     (-H_21, 1 - H_22)
        Ax_1 = x_1.copy()
        Ax_2 = x_2.copy()
        operands_11 = [H_11_left, H_11_right, x_1, d1x_d1x]
        operands_12 = [H_12_left, H_12_right, x_2, d1x_d1y]
        operands_21 = [H_12_left, H_12_right, x_1, d1x_d1y]
        operands_22 = [H_22_left, H_22_right, x_2, d1y_d1y]

        o = OptimizerOptions(path=[(0, 2), (1, 2), (0, 1)])


        if 'cg_d' not in self.networks.keys():
            self.networks['cg_d'] = [None]*self.n_bits
            self.networks['cg_12'] = [None]*self.n_bits
            self.networks['cg_21'] = [None]*self.n_bits

        if self.networks[f'cg_d'][pos] is None:
            self.networks['cg_d'][pos] = Network('umd, reD, upr, mpeP->dPD', *operands_11, options=options)
            path, info = self.networks['cg_d'][pos].contract_path(optimize=o)
            self.networks['cg_12'][pos] = Network('umd, reD, upr, mpeP->dPD', *operands_12, options=options) 
            path, info = self.networks['cg_12'][pos].contract_path(optimize=o)
            self.networks['cg_21'][pos] = Network('umd, reD, dPD, mpeP->upr', *operands_21, options=options) 
            path, info = self.networks['cg_21'][pos].contract_path(optimize=o)
        else:
            self.networks['cg_d'][pos].reset_operands(*operands_11)
            self.networks['cg_12'][pos].reset_operands(*operands_12)
            self.networks['cg_21'][pos].reset_operands(*operands_21)
        
        Ax_1 -= self.networks['cg_d'][pos].contract() + self.networks['cg_12'][pos].contract(release_workspace=True)
        Ax_2 -= self.networks['cg_21'][pos].contract(release_workspace=True)

        self.networks['cg_d'][pos].reset_operands(*operands_22)
        Ax_2 -= self.networks['cg_d'][pos].contract(release_workspace=True)

        # self.networks['cg_d'][pos].workspace_ptr = None
        # self.networks['cg_12'][pos].workspace_ptr = None
        # self.networks['cg_21'][pos].workspace_ptr = None

        return Ax_1, Ax_2


    # conjugate gradient algorithm in MPS form
    def solve_LS_cg(self, H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y, b_1, b_2, pos, options=None):
        Ax_1, Ax_2 = self.Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y, pos, options)
        r_1 = b_1 - Ax_1
        r_2 = b_2 - Ax_2
        p_1 = r_1
        p_2 = r_2

        o = OptimizerOptions(path=[(0, 1)])
        if 'residual' not in self.networks.keys():
            self.networks['residual'] = [None]*self.n_bits

        if self.networks['residual'][pos] is None:
            self.networks['residual'][pos] = Network('apb, apb->', r_1, r_1, options=options) 
            path, info = self.networks['residual'][pos].contract_path(optimize=o)
        else:
            self.networks['residual'][pos].reset_operands(r_1, r_1)
        r_r = self.networks['residual'][pos].contract()
        self.networks['residual'][pos].reset_operands(r_2, r_2)
        r_r += self.networks['residual'][pos].contract()

        iter = 0
        # n = 2
        # for s in b_1.shape:
        #     n *= s
        # max_iter = 10*n
        max_iter = 100
        while r_r > 1e-5 and iter < max_iter:
            iter += 1

            Ap_1, Ap_2 = self.Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, p_1, p_2, d1x_d1x, d1x_d1y, d1y_d1y, pos, options)
            self.networks['residual'][pos].reset_operands(p_1, Ap_1)
            pAp_1 = self.networks['residual'][pos].contract()
            self.networks['residual'][pos].reset_operands(p_2, Ap_2)
            pAp_2 = self.networks['residual'][pos].contract()
            alpha = r_r / (pAp_1 + pAp_2)

            x_1 = x_1 + alpha * p_1
            x_2 = x_2 + alpha * p_2

            Ax_1, Ax_2 = self.Ax(H_11_left, H_11_right, H_12_left, H_12_right, H_22_left, H_22_right, x_1, x_2, d1x_d1x, d1x_d1y, d1y_d1y, pos, options)
            r_new_1 = b_1 - Ax_1
            r_new_2 = b_2 - Ax_2
            
            self.networks['residual'][pos].reset_operands(r_new_1, r_new_1)
            r_new_r_new = self.networks['residual'][pos].contract()
            self.networks['residual'][pos].reset_operands(r_new_2, r_new_2)
            r_new_r_new += self.networks['residual'][pos].contract(release_workspace=True)
            beta = r_new_r_new / r_r

            p_1 = r_new_1 + beta * p_1
            p_2 = r_new_2 + beta * p_2

            r_r = r_new_r_new
        
        # print(iter, r_r)
        # self.networks['residual'][pos].workspace_ptr = None

        return x_1, x_2


    # linear system solver via matrix inversion in MPS form
    def solve_LS_inv(self, H_11, H_12, H_22, b_1, b_2):
        shape = b_1.shape
        dim = 1
        for d in shape:
            dim *= d
        b_1 = b_1.flatten().get()
        b_2 = b_2.flatten().get()
        H_11 = H_11.reshape((dim, dim)).get()
        H_12 = H_12.reshape((dim, dim)).get()
        H_22 = H_22.reshape((dim, dim)).get()
        
        H = np.block([[H_11, H_12.T], [H_12, H_22]])
        A = np.eye(len(H)) - H
        b = np.concatenate((b_1, b_2))
        
        x = np.linalg.solve(A,b)
        U_new, V_new = np.array_split(x, 2)

        return cp.asarray(U_new.reshape(shape)), cp.asarray(V_new.reshape(shape))


    # linear system solver via scipy.cg in MPS form
    def solve_LS_cg_scipy(self, H_11, H_12, H_22, x_1, x_2, b_1, b_2):
        shape = x_1.shape
        dim = 1
        for d in shape:
            dim *= d
        b_1 = b_1.flatten().get()
        b_2 = b_2.flatten().get()
        x_1 = x_1.flatten().get()
        x_2 = x_2.flatten().get()
        H_11 = H_11.reshape((dim, dim)).get()
        H_12 = H_12.reshape((dim, dim)).get()
        H_22 = H_22.reshape((dim, dim)).get()
        
        H = np.block([[H_11, H_12.T], [H_12, H_22]])
        A = np.eye(len(H)) - H
        b = np.concatenate((b_1, b_2))
        x = np.concatenate((x_1, x_2))
        # print(np.linalg.cond(A))
        x_sol = cg(A, b, x)
        U_new, V_new = np.array_split(x_sol[0], 2)

        return cp.asarray(U_new.reshape(shape)), cp.asarray(V_new.reshape(shape))


    # helper function to compute convection-diffusion terms
    def left_right_A_W(self, left_tn, right_tn, A_t, W_t, pos, extra='', options=None, contract_string='umd, upr, mpeP, reD->dPD'):
        operands = [left_tn, A_t, W_t, right_tn]
        o = OptimizerOptions(path=[(0, 1), (0, 2), (0, 1)])

        if f'l_r_A_W{extra}' not in self.networks.keys():
            self.networks[f'l_r_A_W{extra}'] = [None]*self.n_bits

        if self.networks[f'l_r_A_W{extra}'][pos] is None:
            self.networks[f'l_r_A_W{extra}'][pos] = Network(contract_string, *operands, options=options)
            path, info = self.networks[f'l_r_A_W{extra}'][pos].contract_path(optimize = o)
        else:
            self.networks[f'l_r_A_W{extra}'][pos].reset_operands(*operands)

        tensor = self.networks[f'l_r_A_W{extra}'][pos].contract(release_workspace=True)
        # self.networks[f'l_r_A_W{extra}'][pos].workspace_ptr = None

        return tensor


    def left_right_W(self, left_tn, right_tn, W_t, options=None, contract_string='umd, mpeP, reD->uprdPD'):
        o = OptimizerOptions(path=[(0, 1), (0, 1)])

        return contract(contract_string, left_tn, W_t, right_tn, options=options, optimize=o)

    def single_time_step(self, dt, U, V, Ax_MPS, Ay_MPS, Bx_MPS, By_MPS, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, solver='cg', options=None):
    
        # ========== TIMING INITIALIZATION ==========
        import time
        timings = {
            'setup_precontract_Ax': 0,
            'mpo_creation': 0,
            'precontract_convdiff': 0,
            'energy_init': 0,
            'sweeps_total': 0,
            'sweep_build_b': 0,
            'sweep_solver': 0,
            'sweep_compression': 0,
            'sweep_updates': 0,
        }
        
        # ========== SECTION 1: PRECONTRACT Ax/Ay ==========
        t_start = time.time()
        U_Ax_left, U_Ax_right = self.get_precontracted_LR_mps_mps(U, Ax_MPS, 0, options)
        V_Ay_left, V_Ay_right = self.get_precontracted_LR_mps_mps(V, Ay_MPS, 0, options)
        timings['setup_precontract_Ax'] = time.time() - t_start
        
        # ========== SECTION 2: MPO CREATION (Hadamard + Multiply) ==========
        t_start = time.time()
        mult_algorithm = {'qr_method': False, 'svd_method': {'partition': 'V', 'max_extent': self.chi_mpo}}
        
        Bx_MPO = self.hadamard_product_MPO(Bx_MPS, options)
        Bxd1x = self.multiply_mpo_mpo(Bx_MPO, d1x, mult_algorithm, options)
        d1xBx = self.multiply_mpo_mpo(d1x, Bx_MPO, mult_algorithm, options)
        By_MPO = self.hadamard_product_MPO(By_MPS, options)
        Byd1y = self.multiply_mpo_mpo(By_MPO, d1y, mult_algorithm, options)
        d1yBy = self.multiply_mpo_mpo(d1y, By_MPO, mult_algorithm, options)
        timings['mpo_creation'] = time.time() - t_start
        
        # ========== SECTION 3: PRECONTRACT CONVECTION-DIFFUSION TERMS ==========
        t_start = time.time()
        # x direction
        U_d2x_Bx_left, U_d2x_Bx_right = self.get_precontracted_LR_mps_mpo(U, d2x, Bx_MPS, 0, '_d', options)
        U_d2y_Bx_left, U_d2y_Bx_right = self.get_precontracted_LR_mps_mpo(U, d2y, Bx_MPS, 0, '_d', options)
        U_Bxd1x_Bx_left, U_Bxd1x_Bx_right = self.get_precontracted_LR_mps_mpo(U, Bxd1x, Bx_MPS, 0, '_f', options)
        U_d1xBx_Bx_left, U_d1xBx_Bx_right = self.get_precontracted_LR_mps_mpo(U, d1xBx, Bx_MPS, 0, '_f', options)
        U_Byd1y_Bx_left, U_Byd1y_Bx_right = self.get_precontracted_LR_mps_mpo(U, Byd1y, Bx_MPS, 0, '_f', options)
        U_d1yBy_Bx_left, U_d1yBy_Bx_right = self.get_precontracted_LR_mps_mpo(U, d1yBy, Bx_MPS, 0, '_f', options)
        
        # y direction
        V_d2x_By_left, V_d2x_By_right = self.get_precontracted_LR_mps_mpo(V, d2x, By_MPS, 0, '_d', options)
        V_d2y_By_left, V_d2y_By_right = self.get_precontracted_LR_mps_mpo(V, d2y, By_MPS, 0, '_d', options)
        V_Bxd1x_By_left, V_Bxd1x_By_right = self.get_precontracted_LR_mps_mpo(V, Bxd1x, By_MPS, 0, '_f', options)
        V_d1xBx_By_left, V_d1xBx_By_right = self.get_precontracted_LR_mps_mpo(V, d1xBx, By_MPS, 0, '_f', options)
        V_Byd1y_By_left, V_Byd1y_By_right = self.get_precontracted_LR_mps_mpo(V, Byd1y, By_MPS, 0, '_f', options)
        V_d1yBy_By_left, V_d1yBy_By_right = self.get_precontracted_LR_mps_mpo(V, d1yBy, By_MPS, 0, '_f', options)
        timings['precontract_convdiff'] = time.time() - t_start
        
        # ========== SECTION 4: ENERGY INITIALIZATION ==========
        t_start = time.time()
        epsilon = 1e-5
        E_0 = 1e-10
        operands = [U[0], U[0]]
        o = OptimizerOptions(path=[(0, 1)])
        if 'norm' not in self.networks.keys():
            self.networks['norm'] = Network('apb, apb->', *operands, options=options)
            path, info = self.networks['norm'].contract_path(optimize = o)
        else:
            self.networks['norm'].reset_operands(*operands)
        
        E_1_U = self.networks['norm'].contract()
        operands = [V[0], V[0]]
        self.networks['norm'].reset_operands(*operands)
        E_1_V = self.networks['norm'].contract(release_workspace=True)
        E_1 = E_1_U + E_1_V
        timings['energy_init'] = time.time() - t_start
        
        # ========== SECTION 5: SWEEP LOOPS ==========
        t_sweeps_start = time.time()
        run = 0
        while np.abs((E_1-E_0)/E_0) > epsilon and run < self.max_sweeps:
            run += 1
            
            # ========== FORWARD SWEEP ==========
            for i in range(self.n_bits-1):
                # --- Build b_1, b_2 (12 contractions) ---
                t_b = time.time()
                operands = [U_Ax_left[i], Ax_MPS[i], U_Ax_right[i]]
                o = OptimizerOptions(path=[(0, 1), (0, 1)])
                
                if 'b' not in self.networks.keys():
                    self.networks['b'] = [None]*self.n_bits
                
                if self.networks['b'][i] is None:
                    self.networks['b'][i] = Network('ud, upr, rD->dpD', *operands, options=options)
                    path, info = self.networks['b'][i].contract_path(optimize = o)
                else:
                    self.networks['b'][i].reset_operands(*operands)
                
                b_1 = self.networks['b'][i].contract()
                
                # 6 left_right_A_W calls for b_1
                b_1 += dt/self.Re * self.left_right_A_W(U_d2x_Bx_left[i], U_d2x_Bx_right[i], Bx_MPS[i], d2x[i], i, '_d', options)
                b_1 += dt/self.Re * self.left_right_A_W(U_d2y_Bx_left[i], U_d2y_Bx_right[i], Bx_MPS[i], d2y[i], i, '_d', options)
                b_1 += -dt/2 * self.left_right_A_W(U_Bxd1x_Bx_left[i], U_Bxd1x_Bx_right[i], Bx_MPS[i], Bxd1x[i], i, '_f', options)
                b_1 += -dt/2 * self.left_right_A_W(U_d1xBx_Bx_left[i], U_d1xBx_Bx_right[i], Bx_MPS[i], d1xBx[i], i, '_f', options)
                b_1 += -dt/2 * self.left_right_A_W(U_Byd1y_Bx_left[i], U_Byd1y_Bx_right[i], Bx_MPS[i], Byd1y[i], i, '_f', options)
                b_1 += -dt/2 * self.left_right_A_W(U_d1yBy_Bx_left[i], U_d1yBy_Bx_right[i], Bx_MPS[i], d1yBy[i], i, '_f', options)
                
                operands = [V_Ay_left[i], Ay_MPS[i], V_Ay_right[i]]
                self.networks['b'][i].reset_operands(*operands)
                b_2 = self.networks['b'][i].contract(release_workspace=True)
                
                # 6 left_right_A_W calls for b_2
                b_2 += dt/self.Re * self.left_right_A_W(V_d2x_By_left[i], V_d2x_By_right[i], By_MPS[i], d2x[i], i, '_d', options)
                b_2 += dt/self.Re * self.left_right_A_W(V_d2y_By_left[i], V_d2y_By_right[i], By_MPS[i], d2y[i], i, '_d', options)
                b_2 += -dt/2 * self.left_right_A_W(V_Bxd1x_By_left[i], V_Bxd1x_By_right[i], By_MPS[i], Bxd1x[i], i, '_f', options)
                b_2 += -dt/2 * self.left_right_A_W(V_d1xBx_By_left[i], V_d1xBx_By_right[i], By_MPS[i], d1xBx[i], i, '_f', options)
                b_2 += -dt/2 * self.left_right_A_W(V_Byd1y_By_left[i], V_Byd1y_By_right[i], By_MPS[i], Byd1y[i], i, '_f', options)
                b_2 += -dt/2 * self.left_right_A_W(V_d1yBy_By_left[i], V_d1yBy_By_right[i], By_MPS[i], d1yBy[i], i, '_f', options)
                timings['sweep_build_b'] += time.time() - t_b
                
                # --- Solve linear system ---
                t_solve = time.time()
                if solver == 'cg':
                    U_new, V_new = self.solve_LS_cg(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], U[i], V[i], self.mu*dt**2 * d1x_d1x[i], self.mu*dt**2 * d1x_d1y[i], self.mu*dt**2 * d1y_d1y[i], b_1, b_2, i, options)
                elif solver == 'inv':
                    H_11 = self.mu*dt**2 * self.left_right_W(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x[i], options)
                    H_12 = self.mu*dt**2 * self.left_right_W(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y[i], options)
                    H_22 = self.mu*dt**2 * self.left_right_W(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y[i], options)
                    U_new, V_new = self.solve_LS_inv(H_11, H_12, H_22, b_1, b_2)
                elif solver == 'scipy.cg':
                    H_11 = self.mu*dt**2 * self.left_right_W(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x[i], options)
                    H_12 = self.mu*dt**2 * self.left_right_W(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y[i], options)
                    H_22 = self.mu*dt**2 * self.left_right_W(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y[i], options)
                    x_1 = U[i].copy()
                    x_2 = V[i].copy()
                    U_new, V_new = self.solve_LS_cg_scipy(H_11, H_12, H_22, x_1, x_2, b_1, b_2)
                else:
                    raise Exception(f"The solver '{solver}' is not known.")
                timings['sweep_solver'] += time.time() - t_solve
                
                # --- Update MPS and shift canonical center ---
                U[i] = U_new
                V[i] = V_new
                
                t_comp = time.time()
                U = self.shift_canonical_center(U, i+1, i, options)
                V = self.shift_canonical_center(V, i+1, i, options)
                timings['sweep_compression'] += time.time() - t_comp
                
                # --- Update precontracted networks (18 calls) ---
                t_upd = time.time()
                U_Ax_left[i+1] = self.update_precontracted_LR_mps_mps(U_Ax_left[i], U[i], Ax_MPS[i], 'L', i, options)
                V_Ay_left[i+1] = self.update_precontracted_LR_mps_mps(V_Ay_left[i], V[i], Ay_MPS[i], 'L', i, options)
                
                U_d2x_Bx_left[i+1] = self.update_precontracted_LR_mps_mpo(U_d2x_Bx_left[i], U[i], d2x[i], Bx_MPS[i], 'L', i, '_d', options)
                U_d2y_Bx_left[i+1] = self.update_precontracted_LR_mps_mpo(U_d2y_Bx_left[i], U[i], d2y[i], Bx_MPS[i], 'L', i, '_d', options)
                U_Bxd1x_Bx_left[i+1] = self.update_precontracted_LR_mps_mpo(U_Bxd1x_Bx_left[i], U[i], Bxd1x[i], Bx_MPS[i], 'L', i, '_f', options)
                U_d1xBx_Bx_left[i+1] = self.update_precontracted_LR_mps_mpo(U_d1xBx_Bx_left[i], U[i], d1xBx[i], Bx_MPS[i], 'L', i, '_f', options)
                U_Byd1y_Bx_left[i+1] = self.update_precontracted_LR_mps_mpo(U_Byd1y_Bx_left[i], U[i], Byd1y[i], Bx_MPS[i], 'L', i, '_f', options)
                U_d1yBy_Bx_left[i+1] = self.update_precontracted_LR_mps_mpo(U_d1yBy_Bx_left[i], U[i], d1yBy[i], Bx_MPS[i], 'L', i, '_f', options)
                
                V_d2x_By_left[i+1] = self.update_precontracted_LR_mps_mpo(V_d2x_By_left[i], V[i], d2x[i], By_MPS[i], 'L', i, '_d', options)
                V_d2y_By_left[i+1] = self.update_precontracted_LR_mps_mpo(V_d2y_By_left[i], V[i], d2y[i], By_MPS[i], 'L', i, '_d', options)
                V_Bxd1x_By_left[i+1] = self.update_precontracted_LR_mps_mpo(V_Bxd1x_By_left[i], V[i], Bxd1x[i], By_MPS[i], 'L', i, '_f', options)
                V_d1xBx_By_left[i+1] = self.update_precontracted_LR_mps_mpo(V_d1xBx_By_left[i], V[i], d1xBx[i], By_MPS[i], 'L', i, '_f', options)
                V_Byd1y_By_left[i+1] = self.update_precontracted_LR_mps_mpo(V_Byd1y_By_left[i], V[i], Byd1y[i], By_MPS[i], 'L', i, '_f', options)
                V_d1yBy_By_left[i+1] = self.update_precontracted_LR_mps_mpo(V_d1yBy_By_left[i], V[i], d1yBy[i], By_MPS[i], 'L', i, '_f', options)
                
                U_d1x_d1x_U_left[i+1] = self.update_precontracted_LR_mps_mpo(U_d1x_d1x_U_left[i], U[i], d1x_d1x[i], U[i], 'L', i, '_dd', options)
                U_d1x_d1y_V_left[i+1] = self.update_precontracted_LR_mps_mpo(U_d1x_d1y_V_left[i], U[i], d1x_d1y[i], V[i], 'L', i, '_ddxy', options)
                V_d1y_d1y_V_left[i+1] = self.update_precontracted_LR_mps_mpo(V_d1y_d1y_V_left[i], V[i], d1y_d1y[i], V[i], 'L', i, '_dd', options)
                timings['sweep_updates'] += time.time() - t_upd
            
            # ========== BACKWARD SWEEP (identical structure) ==========
            for i in range(self.n_bits-1, 0, -1):
                t_b = time.time()
                operands = [U_Ax_left[i], Ax_MPS[i], U_Ax_right[i]]
                if self.networks['b'][i] is None:
                    self.networks['b'][i] = Network('ud, upr, rD->dpD', *operands, options=options)
                    path, info = self.networks['b'][i].contract_path(optimize = o)
                else:
                    self.networks['b'][i].reset_operands(*operands)
                b_1 = self.networks['b'][i].contract()
                
                b_1 += dt/self.Re * self.left_right_A_W(U_d2x_Bx_left[i], U_d2x_Bx_right[i], Bx_MPS[i], d2x[i], i, '_d', options)
                b_1 += dt/self.Re * self.left_right_A_W(U_d2y_Bx_left[i], U_d2y_Bx_right[i], Bx_MPS[i], d2y[i], i, '_d', options)
                b_1 += -dt/2 * self.left_right_A_W(U_Bxd1x_Bx_left[i], U_Bxd1x_Bx_right[i], Bx_MPS[i], Bxd1x[i], i, '_f', options)
                b_1 += -dt/2 * self.left_right_A_W(U_d1xBx_Bx_left[i], U_d1xBx_Bx_right[i], Bx_MPS[i], d1xBx[i], i, '_f', options)
                b_1 += -dt/2 * self.left_right_A_W(U_Byd1y_Bx_left[i], U_Byd1y_Bx_right[i], Bx_MPS[i], Byd1y[i], i, '_f', options)
                b_1 += -dt/2 * self.left_right_A_W(U_d1yBy_Bx_left[i], U_d1yBy_Bx_right[i], Bx_MPS[i], d1yBy[i], i, '_f', options)
                
                operands = [V_Ay_left[i], Ay_MPS[i], V_Ay_right[i]]
                self.networks['b'][i].reset_operands(*operands)
                b_2 = self.networks['b'][i].contract(release_workspace=True)
                
                b_2 += dt/self.Re * self.left_right_A_W(V_d2x_By_left[i], V_d2x_By_right[i], By_MPS[i], d2x[i], i, '_d', options)
                b_2 += dt/self.Re * self.left_right_A_W(V_d2y_By_left[i], V_d2y_By_right[i], By_MPS[i], d2y[i], i, '_d', options)
                b_2 += -dt/2 * self.left_right_A_W(V_Bxd1x_By_left[i], V_Bxd1x_By_right[i], By_MPS[i], Bxd1x[i], i, '_f', options)
                b_2 += -dt/2 * self.left_right_A_W(V_d1xBx_By_left[i], V_d1xBx_By_right[i], By_MPS[i], d1xBx[i], i, '_f', options)
                b_2 += -dt/2 * self.left_right_A_W(V_Byd1y_By_left[i], V_Byd1y_By_right[i], By_MPS[i], Byd1y[i], i, '_f', options)
                b_2 += -dt/2 * self.left_right_A_W(V_d1yBy_By_left[i], V_d1yBy_By_right[i], By_MPS[i], d1yBy[i], i, '_f', options)
                timings['sweep_build_b'] += time.time() - t_b
                
                t_solve = time.time()
                if solver == 'cg':
                    U_new, V_new = self.solve_LS_cg(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], U[i], V[i], self.mu*dt**2 * d1x_d1x[i], self.mu*dt**2 * d1x_d1y[i], self.mu*dt**2 * d1y_d1y[i], b_1, b_2, i, options)
                elif solver == 'inv':
                    H_11 = self.mu*dt**2 * self.left_right_W(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x[i], options)
                    H_12 = self.mu*dt**2 * self.left_right_W(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y[i], options)
                    H_22 = self.mu*dt**2 * self.left_right_W(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y[i], options)
                    U_new, V_new = self.solve_LS_inv(H_11, H_12, H_22, b_1, b_2)
                elif solver == 'scipy.cg':
                    H_11 = self.mu*dt**2 * self.left_right_W(U_d1x_d1x_U_left[i], U_d1x_d1x_U_right[i], d1x_d1x[i], options)
                    H_12 = self.mu*dt**2 * self.left_right_W(U_d1x_d1y_V_left[i], U_d1x_d1y_V_right[i], d1x_d1y[i], options)
                    H_22 = self.mu*dt**2 * self.left_right_W(V_d1y_d1y_V_left[i], V_d1y_d1y_V_right[i], d1y_d1y[i], options)
                    x_1 = U[i].copy()
                    x_2 = V[i].copy()
                    U_new, V_new = self.solve_LS_cg_scipy(H_11, H_12, H_22, x_1, x_2, b_1, b_2)
                else:
                    raise Exception(f"The solver '{solver}' is not known.")
                timings['sweep_solver'] += time.time() - t_solve
                
                U[i] = U_new
                V[i] = V_new
                
                t_comp = time.time()
                U = self.shift_canonical_center(U, i-1, i, options)
                V = self.shift_canonical_center(V, i-1, i, options)
                timings['sweep_compression'] += time.time() - t_comp
                
                t_upd = time.time()
                U_Ax_right[i-1] = self.update_precontracted_LR_mps_mps(U_Ax_right[i], U[i], Ax_MPS[i], 'R', i, options)
                V_Ay_right[i-1] = self.update_precontracted_LR_mps_mps(V_Ay_right[i], V[i], Ay_MPS[i], 'R', i, options)
                
                U_d2x_Bx_right[i-1] = self.update_precontracted_LR_mps_mpo(U_d2x_Bx_right[i], U[i], d2x[i], Bx_MPS[i], 'R', i, '_d', options)
                U_d2y_Bx_right[i-1] = self.update_precontracted_LR_mps_mpo(U_d2y_Bx_right[i], U[i], d2y[i], Bx_MPS[i], 'R', i, '_d', options)
                U_Bxd1x_Bx_right[i-1] = self.update_precontracted_LR_mps_mpo(U_Bxd1x_Bx_right[i], U[i], Bxd1x[i], Bx_MPS[i], 'R', i, '_f', options)
                U_d1xBx_Bx_right[i-1] = self.update_precontracted_LR_mps_mpo(U_d1xBx_Bx_right[i], U[i], d1xBx[i], Bx_MPS[i], 'R', i, '_f', options)
                U_Byd1y_Bx_right[i-1] = self.update_precontracted_LR_mps_mpo(U_Byd1y_Bx_right[i], U[i], Byd1y[i], Bx_MPS[i], 'R', i, '_f', options)
                U_d1yBy_Bx_right[i-1] = self.update_precontracted_LR_mps_mpo(U_d1yBy_Bx_right[i], U[i], d1yBy[i], Bx_MPS[i], 'R', i, '_f', options)
                
                V_d2x_By_right[i-1] = self.update_precontracted_LR_mps_mpo(V_d2x_By_right[i], V[i], d2x[i], By_MPS[i], 'R', i, '_d', options)
                V_d2y_By_right[i-1] = self.update_precontracted_LR_mps_mpo(V_d2y_By_right[i], V[i], d2y[i], By_MPS[i], 'R', i, '_d', options)
                V_Bxd1x_By_right[i-1] = self.update_precontracted_LR_mps_mpo(V_Bxd1x_By_right[i], V[i], Bxd1x[i], By_MPS[i], 'R', i, '_f', options)
                V_d1xBx_By_right[i-1] = self.update_precontracted_LR_mps_mpo(V_d1xBx_By_right[i], V[i], d1xBx[i], By_MPS[i], 'R', i, '_f', options)
                V_Byd1y_By_right[i-1] = self.update_precontracted_LR_mps_mpo(V_Byd1y_By_right[i], V[i], Byd1y[i], By_MPS[i], 'R', i, '_f', options)
                V_d1yBy_By_right[i-1] = self.update_precontracted_LR_mps_mpo(V_d1yBy_By_right[i], V[i], d1yBy[i], By_MPS[i], 'R', i, '_f', options)
                
                U_d1x_d1x_U_right[i-1] = self.update_precontracted_LR_mps_mpo(U_d1x_d1x_U_right[i], U[i], d1x_d1x[i], U[i], 'R', i, '_dd', options)
                U_d1x_d1y_V_right[i-1] = self.update_precontracted_LR_mps_mpo(U_d1x_d1y_V_right[i], U[i], d1x_d1y[i], V[i], 'R', i, '_ddxy', options)
                V_d1y_d1y_V_right[i-1] = self.update_precontracted_LR_mps_mpo(V_d1y_d1y_V_right[i], V[i], d1y_d1y[i], V[i], 'R', i, '_dd', options)
                timings['sweep_updates'] += time.time() - t_upd
            
            # Energy convergence check
            E_0 = E_1
            operands = [U[0], U[0]]
            self.networks['norm'].reset_operands(*operands)
            E_1_U = self.networks['norm'].contract()
            operands = [V[0], V[0]]
            self.networks['norm'].reset_operands(*operands)
            E_1_V = self.networks['norm'].contract(release_workspace=True)
            E_1 = E_1_U + E_1_V
            print(f"Run: {run}, Diff: {(E_1-E_0)/E_0}, E_0: {E_0}, E_1: {E_1}", end='\r')
        
        timings['sweeps_total'] = time.time() - t_sweeps_start
        
        # ========== PRINT TIMING BREAKDOWN ==========
        total_time = time.time() - (t_sweeps_start - timings['setup_precontract_Ax'] - timings['mpo_creation'] - timings['precontract_convdiff'] - timings['energy_init'])
        print(f"\n\n========== TIMING BREAKDOWN ==========")
        print(f"{'Section':<30} {'Time (s)':<10} {'% of Total':<10}")
        print("-" * 50)
        for key, val in timings.items():
            pct = (val / total_time * 100) if total_time > 0 else 0
            print(f"{key:<30} {val:>8.3f}   {pct:>6.1f}%")
        print("-" * 50)
        print(f"{'TOTAL':<30} {total_time:>8.3f}")
        print(f"========================================\n")
        
        print(f"Run: {run}, Diff: {(E_1-E_0)/E_0}, E_0: {E_0}, E_1: {E_1}")
        print(f"Time evolution finished in {round(total_time, 2)} seconds.")
        
        return self.copy_tn(U), self.copy_tn(V)



    def plot(self, U, V, time=-1, full=False, save_path=None, show=False):
        # plot velocity field given as MPS
        u = self.convert_to_VF2D(self.convert_cupy_to_MPS(U))
        v = self.convert_to_VF2D(self.convert_cupy_to_MPS(V))  

        # Genaral parameters
        x = np.linspace(0, 1-self.dx, self.N)
        y = np.linspace(0, 1-self.dx, self.N)
        Y, X = np.meshgrid(y, x)
        n_s = 2**(self.n_bits-4)                  # Plot N/n_s number of arrows

        plt.figure()
        plt.contourf(X, Y, Dx(v, self.dx)-Dy(u, self.dx), 100, cmap="seismic")
        plt.colorbar()
        if full:
            plt.quiver(X, Y, u, v, color="black")
        else:
            plt.quiver(X[::n_s, ::n_s], Y[::n_s, ::n_s], u[::n_s, ::n_s], v[::n_s, ::n_s], color="black")
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.title(f"Time: {round(time, 5)}")
        if save_path is not None:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()


    # Free all network objects
    def free_networks(self, n_dict):
        for key, el in n_dict.items():
            if isinstance(el, dict):
                self.free_networks(el)
            elif isinstance(el, list):
                for el_ in el:
                    if el_ is not None:
                        el_.free()
            else:
                el.free()


    def multiply_scalar_mps(self, scalar, mps):
        mps[0] = mps[0]*scalar

        return mps
    
    
    def add_mps(self, mps_1, mps_2):
        result_mps = []
        
        for i, (tensor_1, tensor_2) in enumerate(zip(mps_1, mps_2)):
            l_1, p_1, r_1 = tensor_1.shape
            l_2, p_2, r_2 = tensor_2.shape
            if i == 0:
                l = l_1
                p = p_1
                r = r_1 + r_2

                new_tensor = cp.zeros((l, p, r), dtype=tensor_1.dtype)
                new_tensor[:l_1, :, :r_1] = tensor_1
                new_tensor[:l_1, :, r_1:] = tensor_2
            elif i == len(mps_1)-1:
                l = l_1 + l_2
                p = p_1
                r = r_1

                new_tensor = cp.zeros((l, p, r), dtype=tensor_1.dtype)
                new_tensor[:l_1, :, :r_1] = tensor_1
                new_tensor[l_1:, :, :r_1] = tensor_2
            else:
                l = l_1 + l_2
                p = p_1
                r = r_1 + r_2
                
                new_tensor = cp.zeros((l, p, r), dtype=tensor_1.dtype)
                new_tensor[:l_1, :, :r_1] = tensor_1
                new_tensor[l_1:, :, r_1:] = tensor_2
                
            result_mps.append(new_tensor)
        
        return result_mps
    

    def add_mps_list(self, mps_list, coeff_list, chi):
        curr_mps = self.copy_tn(mps_list[0])
        curr_mps[0] *= coeff_list[0]
        for i in range(1, len(mps_list)):
            new_mps = self.copy_tn(mps_list[i])
            new_mps[0] *= coeff_list[i]
            curr_mps = self.add_mps(curr_mps, new_mps)
            curr_mps = self.compress_mps(curr_mps, chi, curr_center=0, options=self.options)
        
        return curr_mps
    

    # time evolution algorithm
    def time_evolution(self):
        n_steps = int(np.ceil((self.T-self.t)/self.dt))    # time steps
        # finite difference operators with 8th order precision
        d1x = Diff_1_8_x_MPO(self.n_bits, self.dx, self.options)
        d1y = Diff_1_8_y_MPO(self.n_bits, self.dx, self.options)
        d2x = Diff_2_8_x_MPO(self.n_bits, self.dx, self.options)
        d2y = Diff_2_8_y_MPO(self.n_bits, self.dx, self.options)

        # finite difference operators with 2nd order precision 
        # d1x = Diff_1_2_x_MPO(n, dx, options)
        # d1y = Diff_1_2_y_MPO(n, dx, options)
        # d2x = Diff_2_2_x_MPO(n, dx, options)
        # d2y = Diff_2_2_y_MPO(n, dx, options)

        mult_algorithm = {'qr_method': False, 'svd_method': {'partition': 'V', 'max_extent': self.chi_mpo}} # 'rel_cutoff':1e-10, 
        d1x_d1x = self.multiply_mpo_mpo(d1x, d1x, mult_algorithm, self.options)
        d1x_d1y = self.multiply_mpo_mpo(d1x, d1y, mult_algorithm, self.options)
        d1y_d1y = self.multiply_mpo_mpo(d1y, d1y, mult_algorithm, self.options)
        
        # bring the orthogonality center to the first tensor
        U = self.canonical_center(self.U_init, 0, self.options)
        V = self.canonical_center(self.V_init, 0, self.options)

        # # initialize precontracted left and right networks
        # U_d1x_d1x_U_left, U_d1x_d1x_U_right = self.get_precontracted_LR_mps_mpo(U, d1x_d1x, U, 0, '_dd', self.options)
        # U_d1x_d1y_V_left, U_d1x_d1y_V_right = self.get_precontracted_LR_mps_mpo(U, d1x_d1y, V, 0, '_ddxy', self.options)
        # V_d1y_d1y_V_left, V_d1y_d1y_V_right = self.get_precontracted_LR_mps_mpo(V, d1y_d1y, V, 0, '_dd', self.options)
    
        comp_time = {}
        print("Simulation begins!")
        for step in range(n_steps):   # for every time step dt
            print(f"Step = {step} - Time = {self.t}", end='\n')
            # if step%5 == 0:
            #     self.plot(U, V, t, save_path=f"/raid/home/q556220/dev/TN_CFD/DJ_2D/cuTensorNet/RK4_DJ/images/{self.n_bits}_{self.chi}_{self.Re}/{t}.png")
            # if self.path is not None and step%(int(n_steps/self.save_number)) == 0:
            if self.path is not None and step%100 == 0:
                np.save(f"{self.path}/u_time_{round(self.t, 5)}.npy", np.array([el.get() for el in U], dtype=object))
                np.save(f"{self.path}/v_time_{round(self.t, 5)}.npy", np.array([el.get() for el in V], dtype=object))

            U_trial = self.copy_tn(U)         # trial velocity state
            V_trial = self.copy_tn(V)         # trial velocity state
            
            if self.meas_comp_time:
                _sync()
                start = time.time()

            # # RK1
            # U, V = self.single_time_step(self.dt, U_trial.copy(), V_trial.copy(), U.copy(), V.copy(), U.copy(), V.copy(), d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, options=self.options)
            
            # # RK2
            # # Midpoint step
            # U_mid, V_mid = self.single_time_step(self.dt/2, U_trial.copy(), V_trial.copy(), U.copy(), V.copy(), U.copy(), V.copy(), d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, options=self.options)
            # # Full step
            # print('')
            # U, V = self.single_time_step(self.dt, U_trial.copy(), V_trial.copy(), U.copy(), V.copy(), U_mid, V_mid, d1x, d1y, d2x, d2y, d1x_d1x, d1x_d1y, d1y_d1y, U_d1x_d1x_U_left, U_d1x_d1x_U_right, U_d1x_d1y_V_left, U_d1x_d1y_V_right, V_d1y_d1y_V_left, V_d1y_d1y_V_right, options=self.options)


            # initialize precontracted left and right networks
            U_d1x_d1x_U_left, U_d1x_d1x_U_right = self.get_precontracted_LR_mps_mpo(U, d1x_d1x, U, 0, '_dd', self.options)
            U_d1x_d1y_V_left, U_d1x_d1y_V_right = self.get_precontracted_LR_mps_mpo(U, d1x_d1y, V, 0, '_ddxy', self.options)
            V_d1y_d1y_V_left, V_d1y_d1y_V_right = self.get_precontracted_LR_mps_mpo(V, d1y_d1y, V, 0, '_dd', self.options)
    
            # RK4
            U1_x, U1_y = self.single_time_step(
                self.dt/6, 
                U_trial, 
                V_trial, 
                self.multiply_scalar_mps(0.25, self.copy_tn(U)), 
                self.multiply_scalar_mps(0.25, self.copy_tn(V)), 
                self.copy_tn(U), 
                self.copy_tn(V),  
                d1x, 
                d1y, 
                d2x, 
                d2y, 
                d1x_d1x, 
                d1x_d1y, 
                d1y_d1y, 
                U_d1x_d1x_U_left, 
                U_d1x_d1x_U_right, 
                U_d1x_d1y_V_left, 
                U_d1x_d1y_V_right, 
                V_d1y_d1y_V_left, 
                V_d1y_d1y_V_right, 
                options=self.options
                )
            print('')
            U2_x, U2_y = self.single_time_step(
                self.dt/3, 
                U_trial, 
                V_trial, 
                self.multiply_scalar_mps(0.25, self.copy_tn(U)), 
                self.multiply_scalar_mps(0.25, self.copy_tn(V)), 
                self.add_mps_list([U1_x, U], [3, 0.25], chi=self.chi),
                self.add_mps_list([U1_y, V], [3, 0.25], chi=self.chi),
                d1x, 
                d1y, 
                d2x, 
                d2y, 
                d1x_d1x, 
                d1x_d1y, 
                d1y_d1y, 
                U_d1x_d1x_U_left, 
                U_d1x_d1x_U_right, 
                U_d1x_d1y_V_left, 
                U_d1x_d1y_V_right, 
                V_d1y_d1y_V_left, 
                V_d1y_d1y_V_right, 
                options=self.options
                )
            print('')
            U3_x, U3_y = self.single_time_step(
                self.dt/3, 
                U_trial, 
                V_trial, 
                self.multiply_scalar_mps(0.25, self.copy_tn(U)), 
                self.multiply_scalar_mps(0.25, self.copy_tn(V)), 
                self.add_mps_list([U2_x, U], [1.5, 5/8], chi=self.chi),
                self.add_mps_list([U2_y, V], [1.5, 5/8], chi=self.chi),
                d1x, 
                d1y, 
                d2x, 
                d2y, 
                d1x_d1x, 
                d1x_d1y, 
                d1y_d1y, 
                U_d1x_d1x_U_left, 
                U_d1x_d1x_U_right, 
                U_d1x_d1y_V_left, 
                U_d1x_d1y_V_right, 
                V_d1y_d1y_V_left, 
                V_d1y_d1y_V_right, 
                options=self.options
                )
            print('')
            U4_x, U4_y = self.single_time_step(
                self.dt/6, 
                U_trial, 
                V_trial, 
                self.multiply_scalar_mps(0.25, self.copy_tn(U)), 
                self.multiply_scalar_mps(0.25, self.copy_tn(V)), 
                self.add_mps_list([U3_x, U], [3, 0.25], chi=self.chi),
                self.add_mps_list([U3_y, V], [3, 0.25], chi=self.chi),
                d1x, 
                d1y, 
                d2x, 
                d2y, 
                d1x_d1x, 
                d1x_d1y, 
                d1y_d1y, 
                U_d1x_d1x_U_left, 
                U_d1x_d1x_U_right, 
                U_d1x_d1y_V_left, 
                U_d1x_d1y_V_right, 
                V_d1y_d1y_V_left, 
                V_d1y_d1y_V_right, 
                options=self.options
                )
            print('')
            U = self.add_mps_list([U1_x, U2_x, U3_x, U4_x], [1, 1, 1, 1], chi=self.chi)
            V = self.add_mps_list([U1_y, U2_y, U3_y, U4_y], [1, 1, 1, 1], chi=self.chi)
            
            print('\n') 
            if self.meas_comp_time:
                _sync()
                end = time.time()
                comp_time[self.t] = end-start

                with open(self.comp_time_path, "w") as outfile: 
                    json.dump(comp_time, outfile)

            self.t += self.dt
        
        # plot(U, V, time=t, save_path=f"{save_path}/final.png", show=False)
        # np.save(f"{save_path}/u_final.npy", np.array([el.get() for el in U], dtype=object))
        # np.save(f"{save_path}/v_final.npy", np.array([el.get() for el in V], dtype=object))
        
        self.free_networks(self.networks)
        self.networks.clear()
    

    # extract occupied gpu memory
    def get_gpu_memory(self, gpu_id=0):

        self.max_sweeps = 1
        # finite difference operators with 8th order precision
        d1x = Diff_1_8_x_MPO(self.n_bits, self.dx, self.options)
        d1y = Diff_1_8_y_MPO(self.n_bits, self.dx, self.options)
        d2x = Diff_2_8_x_MPO(self.n_bits, self.dx, self.options)
        d2y = Diff_2_8_y_MPO(self.n_bits, self.dx, self.options)

        mult_algorithm = {'qr_method': False, 'svd_method': {'partition': 'V', 'max_extent': self.chi_mpo}} # 'rel_cutoff':1e-10, 
        d1x_d1x = self.multiply_mpo_mpo(d1x, d1x, mult_algorithm, self.options)
        d1x_d1y = self.multiply_mpo_mpo(d1x, d1y, mult_algorithm, self.options)
        d1y_d1y = self.multiply_mpo_mpo(d1y, d1y, mult_algorithm, self.options)
        
        # bring the orthogonality center to the first tensor
        U = self.canonical_center(self.U_init, 0, self.options)
        V = self.canonical_center(self.V_init, 0, self.options)

        # initialize precontracted left and right networks
        U_d1x_d1x_U_left, U_d1x_d1x_U_right = self.get_precontracted_LR_mps_mpo(U, d1x_d1x, U, 0, '_dd', self.options)
        U_d1x_d1y_V_left, U_d1x_d1y_V_right = self.get_precontracted_LR_mps_mpo(U, d1x_d1y, V, 0, '_ddxy', self.options)
        V_d1y_d1y_V_left, V_d1y_d1y_V_right = self.get_precontracted_LR_mps_mpo(V, d1y_d1y, V, 0, '_dd', self.options)

        U_trial = self.copy_tn(U)         # trial velocity state
        V_trial = self.copy_tn(V)         # trial velocity state

        # RK4
        U1_x, U1_y = self.single_time_step(
            self.dt/6, 
            U_trial, 
            V_trial, 
            self.multiply_scalar_mps(0.25, self.copy_tn(U)), 
            self.multiply_scalar_mps(0.25, self.copy_tn(V)), 
            self.copy_tn(U), 
            self.copy_tn(V),  
            d1x, 
            d1y, 
            d2x, 
            d2y, 
            d1x_d1x, 
            d1x_d1y, 
            d1y_d1y, 
            U_d1x_d1x_U_left, 
            U_d1x_d1x_U_right, 
            U_d1x_d1y_V_left, 
            U_d1x_d1y_V_right, 
            V_d1y_d1y_V_left, 
            V_d1y_d1y_V_right, 
            options=self.options
            )
        print('')
        U2_x = self.copy_tn(U1_x)
        U2_y = self.copy_tn(U1_x)
        U3_x = self.copy_tn(U1_x)
        U3_y = self.copy_tn(U1_x)
        U4_x = self.copy_tn(U1_x)
        U4_y = self.copy_tn(U1_x)
        
        U = self.add_mps_list([U1_x, U2_x, U3_x, U4_x], [1, 1, 1, 1], chi=self.chi)
        V = self.add_mps_list([U1_y, U2_y, U3_y, U4_y], [1, 1, 1, 1], chi=self.chi)
        
        print('\n') 


        gpu_mem = float(get_gpu_memory()[gpu_id])
        
        self.free_networks(self.networks)
        self.networks.clear()

        return gpu_mem
            

    def build_initial_fields(self, y_min=0.4, y_max=0.6, h=1/200, u_max=1):
        # Generate initial fields
        U, V = self.initial_fields(y_min, y_max, h, u_max) 

        # Rescale into non-dimensional units
        U = U/u_max
        V = V/u_max

        # Convert them to MPS form
        MPS_U = self.convert_to_MPS2D(U, self.chi)
        MPS_V = self.convert_to_MPS2D(V, self.chi)

        # Tranform into quimb MPS form
        MPS_U_cupy= self.convert_MPS_to_cupy(MPS_U, 4)
        MPS_V_cupy = self.convert_MPS_to_cupy(MPS_V, 4)

        self.U_init = MPS_U_cupy
        self.V_init = MPS_V_cupy

        print("Initialized Fields")

    def initial_fields_taylor(self, u_max=1.0):
        """
        Generates Taylor-Green Vortex (TGV) fields.
        Domain: [0, L] x [0, L]
        Formula:
          u =  u0 * sin(2*pi*x) * cos(2*pi*y)
          v = -u0 * cos(2*pi*x) * sin(2*pi*y)
        """
        # 1. Create Grid (indexing='ij' ensures Axis 0 is X, Axis 1 is Y)
        x = np.linspace(0, self.L - self.dx, self.N)
        y = np.linspace(0, self.L - self.dx, self.N)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # 2. Calculate Wavenumber (for periodic domain L)
        k = 2 * np.pi / self.L

        # 3. TGV Analytic Solution
        U =  u_max * np.sin(k * X) * np.cos(k * Y)
        V = -u_max * np.cos(k * X) * np.sin(k * Y)

        return U, V
    
    def build_initial_fields_taylor(self, u_max=1.0):
        print(f"Initializing Taylor-Green Vortex (TGV) for N={self.N}...")

        # 1. Generate TGV Fields
        U, V = self.initial_fields_taylor(u_max=u_max)

        # 2. Rescale (Optional, usually kept 1.0 for TGV)
        # Note: If u_max=1, this effectively does nothing, which is fine.
        U = U / u_max
        V = V / u_max

        # 3. Convert to MPS
        # TGV is Rank-1, so this conversion will be perfect (no truncation error).
        MPS_U = self.convert_to_MPS2D(U, self.chi)
        MPS_V = self.convert_to_MPS2D(V, self.chi)

        # 4. Transform to CuPy/Quimb format
        # (Assuming your convert_MPS_to_cupy handles the device transfer)
        self.U_init = self.convert_MPS_to_cupy(MPS_U, 4)
        self.V_init = self.convert_MPS_to_cupy(MPS_V, 4)

        print("✅ Initialized TGV Fields Successfully.")
    

    def build_dummy_fields(self):
        # Generate initial fields
        MPS_U_cupy = []
        MPS_V_cupy = []

        def left(pos, dim, n):
            if pos <= int(n/2):
                return (2**dim)**pos
            else:
                return (2**dim)**(n-pos)
        
        def right(pos, dim, n):
            if pos < int(n/2):
                return (2**dim)**(pos+1)
            else:
                return (2**dim)**(n-pos-1)
        
        def left_right(pos, dim, n, chi):
            l = left(pos, dim, n)
            if l > chi:
                l = chi
            r = right(pos, dim, n)
            if r > chi:
                r = chi
            
            return l, r

        for i in range(self.n_bits):
            l, r = left_right(i, 2, self.n_bits, self.chi)
            MPS_U_cupy.append(cp.random.random((l, 2**2, r)))
            MPS_V_cupy.append(cp.random.random((l, 2**2, r)))

        self.U_init = MPS_U_cupy
        self.V_init = MPS_V_cupy

        print("Initialized Dummy Fields")

    
    def set_initial_fields(self, U, V, u_max=1):
        # Rescale into non-dimensional units
        U = U/u_max
        V = V/u_max

        # Convert them to MPS form
        MPS_U = self.convert_to_MPS2D(U, self.chi)
        MPS_V = self.convert_to_MPS2D(V, self.chi)

        # Tranform into quimb MPS form
        MPS_U_cupy= self.convert_MPS_to_cupy(MPS_U, 4)
        MPS_V_cupy = self.convert_MPS_to_cupy(MPS_V, 4)

        self.U_init = MPS_U_cupy
        self.V_init = MPS_V_cupy

        print("Initialized Fields")

    
    def set_initial_MPSs(self, u_mps, v_mps, t=0):
        # Rescale into non-dimensional units

        self.U_init = [cp.asarray(tensor) for tensor in u_mps]
        self.V_init = [cp.asarray(tensor) for tensor in v_mps]
        self.t = t

        print("Initialized Fields")
    
    def set_initial_MPS_from_npy(self, u_path, v_path):
        import numpy as np

        u_arr = np.load(u_path, allow_pickle=True)
        v_arr = np.load(v_path, allow_pickle=True)

        assert len(u_arr) == self.n_bits, f"u_arr length {len(u_arr)} != n_bits {self.n_bits}"
        assert len(v_arr) == self.n_bits, f"v_arr length {len(v_arr)} != n_bits {self.n_bits}"

        mps_u_cupy = []
        for core in u_arr:
            # core is already (l, p, r), just move to CuPy
            mps_u_cupy.append(cp.asarray(core))

        mps_v_cupy = []
        for core in v_arr:
            mps_v_cupy.append(cp.asarray(core))

        self.U_init = mps_u_cupy
        self.V_init = mps_v_cupy
        print("Initialized MPS_U and MPS_V from saved MPS .npy files (cores already 3D)")



    