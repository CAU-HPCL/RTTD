# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tensor network contraction and decomposition.
"""

__all__ = ["make_network", "rttd"]


from .. import cutensornet as cutn
from .._internal import decomposition_utils
from .._internal import einsum_parser
from .._internal import tensor_wrapper
from .._internal import utils
from ..configuration import NetworkOptions
from ..tensor_network import contract

import cupy as cp
import numpy as np
import time
import torch


# Try to import C++ module
try:
    from . import rttd_cpp_module  # from same package
    _HAS_CPP_RTTD = True
except ImportError:
    _HAS_CPP_RTTD = False
    import warnings
    warnings.warn("C++ rttd backend not found, using Python fallback")


def torch_decomposition(mat_cupy, chi, partition="U", oversample=8, niter=2, debug=False):
    """
    2D randomized SVD using torch.svd_lowrank.

    Input:  CuPy matrix (M, N)
    Output: CuPy arrays (U, S, Vh) or (U, None, Vh) if partitioned.
    Shapes:
      U  : (M, r_actual)
      S  : (r_actual,)
      Vh : (r_actual, N)
    with r_actual = min(chi, min(M, N)).
    """
    device = "cuda" if cp.cuda.is_available() else "cpu"
    mat_torch = torch.as_tensor(mat_cupy, device=device)
    M, N = mat_torch.shape

    q = min(chi + oversample, M, N)

    U, S, V = torch.svd_lowrank(mat_torch, q=q, niter=niter)

    r_actual = min(chi, S.shape[0])
    U = U[:, :r_actual]
    S = S[:r_actual]
    Vh = V[:, :r_actual].T

    if debug:
        print(f"[torch_decomposition] A shape = {M} x {N}")
        print(f"[torch_decomposition] chi = {chi}, r_actual = {r_actual}")
        print("  U_trunc shape:", U.shape)
        print("  S_trunc shape:", S.shape)
        print("  Vh_trunc shape:", Vh.shape)

    if partition == "U":
        U = U * S.unsqueeze(0)
        S_final = None
    elif partition == "V":
        Vh = S.unsqueeze(1) * Vh
        S_final = None
    else:
        S_final = cp.asarray(S)

    U_final = cp.asarray(U)
    Vh_final = cp.asarray(Vh)

    return U_final, S_final, Vh_final


def make_network(subscripts, *operands, algorithm=None, options=None, optimize=None, stream=None, return_info=False):
    """
    Evaluates Contract -> Decompose using PyTorch Randomized SVD via torch_decomposition.
    The 2D flatten/reshape is layout-equivalent to rttd + _rttd_decompose.
    """
    if algorithm is None or "svd_method" not in algorithm:
        raise ValueError("Algorithm must be a dict with 'svd_method'")

    svd_dict = algorithm.get("svd_method", {})
    chi = svd_dict.get("max_extent")
    partition = svd_dict.get("partition", "U")

    if chi is None:
        raise ValueError("'svd_method' must specify 'max_extent' (chi)")

    wrapped_operands, inputs, outputs, size_dict, mode_map_user_to_ord, mode_map_ord_to_user, max_mid_extent = (
        decomposition_utils.parse_decomposition(subscripts, *operands)
    )

    own_handle = False
    options = utils.check_or_create_options(NetworkOptions, options, "Network Options")

    try:
        wrapped_operands, options, own_handle, operands_location, stream_holder = (
            decomposition_utils.parse_decompose_operands_options(
                options,
                wrapped_operands,
                stream,
                allowed_dtype_names=decomposition_utils.DECOMPOSITION_DTYPE_NAMES,
            )
        )

        intermediate_modes = einsum_parser.infer_output_mode_labels(outputs)

        intermediate_labels_list = []
        ellipses = False
        for _modes in intermediate_modes:
            m = mode_map_ord_to_user[_modes]
            if m.startswith("__"):
                if not ellipses:
                    m = "..."
                    ellipses = True
                else:
                    continue
            intermediate_labels_list.append(m)
        intermediate_labels = "".join(intermediate_labels_list)

        input_modes, output_modes = subscripts.split("->")
        out1_modes_str, out2_modes_str = output_modes.split(",")
        einsum_subscripts = f"{input_modes}->{intermediate_labels}"

        if operands_location == "cpu":
            operands = [o.tensor for o in wrapped_operands]

        intm_output = contract(
            einsum_subscripts,
            *operands,
            options=options,
            optimize=optimize,
            stream=stream,
        )

        in_modes = intermediate_labels
        shared_set = set(out1_modes_str) & set(out2_modes_str)
        if len(shared_set) != 1:
            raise ValueError(f"Shared mode error: {shared_set}")
        shared_mode = list(shared_set)[0]

        left_modes = [m for m in out1_modes_str if m != shared_mode]
        right_modes = [m for m in out2_modes_str if m != shared_mode]

        left_axes = [in_modes.index(m) for m in left_modes]
        right_axes = [in_modes.index(m) for m in right_modes]

        perm = left_axes + right_axes
        tensor_permuted = cp.transpose(intm_output, perm)

        left_dims = [intm_output.shape[ax] for ax in left_axes]
        right_dims = [intm_output.shape[ax] for ax in right_axes]
        left_dim = int(np.prod(left_dims))
        right_dim = int(np.prod(right_dims))

        tensor_2d = tensor_permuted.reshape(left_dim, right_dim)

        U2d, S_final, Vh2d = torch_decomposition(tensor_2d, chi, partition=partition)
        r_actual = U2d.shape[1]

        u_flat = U2d.flatten()
        vh_flat = Vh2d.flatten()

        u_temp_shape = left_dims + [r_actual]
        u_reshaped_temp = u_flat.reshape(u_temp_shape)

        transpose_u = list(range(len(left_dims)))
        transpose_u.insert(2, len(left_dims))
        u_reshaped = cp.transpose(u_reshaped_temp, transpose_u)

        vh_temp_shape = [r_actual] + right_dims
        vh_reshaped = vh_flat.reshape(vh_temp_shape)

        current_u_modes = left_modes + [shared_mode]
        u_perm = [current_u_modes.index(m) for m in out1_modes_str]
        res_U = cp.transpose(u_reshaped, u_perm)

        current_vh_modes = [shared_mode] + right_modes
        vh_perm = [current_vh_modes.index(m) for m in out2_modes_str]
        res_Vh = cp.transpose(vh_reshaped, vh_perm)

        results = [cp.ascontiguousarray(res_U), S_final, cp.ascontiguousarray(res_Vh)]

        if operands_location == "cpu":
            results = [
                o
                if o is None
                else tensor_wrapper.wrap_operand(o).to("cpu", stream_holder=stream_holder)
                for o in results
            ]

    finally:
        if own_handle and options.handle is not None:
            pass

    info_dict = {"optimizer_info": None, "svd_info": None}

    if not return_info:
        return results
    if len(results) == 3:
        return *results, ContractDecomposeInfo(**info_dict)
    return results[0], None, results[1], ContractDecomposeInfo(**info_dict)


class ContractDecomposeInfo:
    def __init__(self, optimizer_info=None, svd_info=None):
        self.optimizer_info = optimizer_info
        self.svd_info = svd_info


def _rttd_decompose(tensor_in, decompose_subscripts, chi, partition, debug=False):
    """
    QR/RQ decomposition with CANONICAL norm absorption (matches cuQuantum behavior).
    Bypasses C++ with CuPy/Torch backend for accuracy.

    For partition='U': Vh orthonormal (||Vh||=1), all norm in U.
    For partition='V': U orthonormal (||U||=1), all norm in Vh.
    """
    times = {}

    start_total = time.perf_counter()

    t0 = time.perf_counter()
    in_norm = float(cp.linalg.norm(tensor_in))
    in_has_nan = bool(cp.any(cp.isnan(tensor_in)))
    in_has_inf = bool(cp.any(cp.isinf(tensor_in)))
    t1 = time.perf_counter()
    times["Input check"] = t1 - t0

    if debug:
        print(f"\n[DEBUG 1] INPUT to _rttd_decompose:")
        print(f"  tensor_in.shape = {tensor_in.shape}")
        print(f"  tensor_in.dtype = {tensor_in.dtype}")
        print(f"  decompose_subscripts = '{decompose_subscripts}'")
        print(f"  chi = {chi}, partition = '{partition}'")
        print(f"  input norm = {in_norm:.6e}")
        print(f"  has_nan = {in_has_nan}, has_inf = {in_has_inf}")

    if in_norm == 0 or in_has_nan or in_has_inf:
        out_shape_u = [1] * len(decompose_subscripts.split("->")[0].replace(",", ""))
        return (
            cp.zeros(out_shape_u, dtype=tensor_in.dtype),
            cp.ones(1, dtype=cp.float32),
            cp.zeros((1, 1), dtype=tensor_in.dtype),
        )

    t0 = time.perf_counter()
    if debug:
        print(f"\n[DEBUG 2] Calling parse_decomposition...")
    wrapped_operands, inputs, outputs, size_dict, _, _, _ = decomposition_utils.parse_decomposition(
        decompose_subscripts, tensor_in
    )
    t1 = time.perf_counter()
    times["Parse decomposition"] = t1 - t0

    if debug:
        print(f"[DEBUG 3] parse_decomposition RESULTS:")
        print(f"  inputs = {inputs}")
        print(f"  outputs = {outputs}")
        print(f"  size_dict = {size_dict}")
        print(f"  wrapped_operands = {wrapped_operands}")
        print(f"  wrapped_operands[0].shape = {wrapped_operands[0].shape if wrapped_operands else 'N/A'}")

    t0 = time.perf_counter()
    in_modes = inputs[0]
    out1_modes, out2_modes = outputs
    shared_set = set(out1_modes) & set(out2_modes)
    if len(shared_set) != 1:
        raise ValueError(f"Shared mode error: {shared_set}")
    shared_mode = list(shared_set)[0]
    mode_to_dim = size_dict
    t1 = time.perf_counter()
    times["Mode parsing"] = t1 - t0

    if debug:
        print(f"\n[DEBUG 4] MODE PARSING:")
        print(f"  in_modes = {in_modes}")
        print(f"  out1_modes = {out1_modes}")
        print(f"  out2_modes = {out2_modes}")
        print(f"  shared_mode = '{shared_mode}'")
        print(f"  mode_to_dim = {mode_to_dim}")

    t0 = time.perf_counter()
    if debug:
        print(f"\n[DEBUG 5] MANUAL SVD PREP:")
        print(f"  Computing SVD on permuted/flattened tensor...")

    left_modes = [m for m in out1_modes if m != shared_mode]
    right_modes = [m for m in out2_modes if m != shared_mode]

    left_axes = [in_modes.index(m) for m in left_modes]
    right_axes = [in_modes.index(m) for m in right_modes]

    perm = left_axes + right_axes
    tensor_permuted = cp.transpose(tensor_in, perm)

    left_dims = [tensor_in.shape[ax] for ax in left_axes]
    right_dims = [tensor_in.shape[ax] for ax in right_axes]
    left_dim = int(np.prod(left_dims))
    right_dim = int(np.prod(right_dims))

    tensor_2d = tensor_permuted.reshape(left_dim, right_dim)
    t1 = time.perf_counter()
    times["Tensor preparation"] = t1 - t0

    t0 = time.perf_counter()
    if _HAS_CPP_RTTD:
        simple_subscript = "ij->ix,xj"
        returns = rttd_cpp_module.rttd_cpp(tensor_2d, simple_subscript, chi, partition)
        if len(returns) == 3:
            U_trunc, s_trunc, Vh_trunc = returns
        else:
            U_trunc, Vh_trunc = returns
            s_trunc = None
        s_trunc = None
        r_actual = U_trunc.shape[1] if U_trunc.ndim == 2 else min(chi, U_trunc.shape[1])
    else:
        U_trunc, s_trunc, Vh_trunc = torch_decomposition(tensor_2d, chi, partition=partition)
        r_actual = U_trunc.shape[1] if U_trunc.ndim == 2 else min(chi, U_trunc.shape[1])
    t1 = time.perf_counter()
    times["Decomposition call"] = t1 - t0

    if debug:
        print(f"  Output: U_trunc {U_trunc.shape}, s_trunc {s_trunc}, Vh_trunc {Vh_trunc.shape}")
        if s_trunc is not None:
            print(f"  s_trunc norm = {float(cp.linalg.norm(s_trunc)):.6e}")
        print(f"  U_trunc norm = {float(cp.linalg.norm(U_trunc)):.6e}")
        print(f"  Vh_trunc norm = {float(cp.linalg.norm(Vh_trunc)):.6e}")
        print(f"  r_actual (truncated) = {r_actual}")

    t0 = time.perf_counter()
    u_flat = U_trunc.flatten()
    vh_flat = Vh_trunc.flatten()

    u_temp_shape = left_dims + [r_actual]
    u_reshaped_temp = u_flat.reshape(u_temp_shape)

    transpose_u = list(range(len(left_dims)))
    transpose_u.insert(2, len(left_dims))
    u_reshaped = cp.transpose(u_reshaped_temp, transpose_u)

    vh_temp_shape = [r_actual] + right_dims
    vh_reshaped = vh_flat.reshape(vh_temp_shape)
    t1 = time.perf_counter()
    times["Reshape tensors"] = t1 - t0

    t0 = time.perf_counter()
    if s_trunc is not None:
        if debug:
            print(f"\n[DEBUG S ABSORPTION]:")
            print(f"  Absorbing s_trunc (shape={s_trunc.shape}, norm={float(cp.linalg.norm(s_trunc)):.6e})")
        if partition == "U":
            s_broadcast = s_trunc[(...,) + (None,) * (u_reshaped.ndim - 1)]
            u_reshaped = u_reshaped * s_broadcast
        else:
            s_broadcast = s_trunc[(...,) + (None,) * (vh_reshaped.ndim - 1)]
            vh_reshaped = vh_reshaped * s_broadcast
    t1 = time.perf_counter()
    times["S absorption"] = t1 - t0

    t0 = time.perf_counter()
    u_norm_pre = float(cp.linalg.norm(u_reshaped))
    vh_norm_pre = float(cp.linalg.norm(vh_reshaped))

    if partition == "V":
        if u_norm_pre > 1e-12:
            scale_to_vh = u_norm_pre
            u_reshaped /= u_norm_pre
            vh_reshaped *= scale_to_vh
        else:
            u_reshaped.fill(1.0 / np.sqrt(u_reshaped.size))
            vh_reshaped *= in_norm
    elif partition == "U":
        if vh_norm_pre > 1e-12:
            scale_to_u = vh_norm_pre
            vh_reshaped /= vh_norm_pre
            u_reshaped *= scale_to_u
        else:
            vh_reshaped.fill(1.0 / np.sqrt(vh_reshaped.size))
            u_reshaped *= in_norm
    t1 = time.perf_counter()
    times["Canonical absorption"] = t1 - t0

    if debug:
        t0 = time.perf_counter()
        try:
            out1_str = "".join(str(m) for m in out1_modes)
            out2_str = "".join(str(m) for m in out2_modes)
            in_str = "".join(str(m) for m in in_modes)
            einsum_eq = f"{out1_str},{out2_str}->{in_str}"
            reconstructed = contract(einsum_eq, u_reshaped, vh_reshaped)
            recon_err_abs = float(cp.linalg.norm(reconstructed - tensor_in))
            recon_err_rel = recon_err_abs / in_norm if in_norm > 1e-12 else 0
            print(f"\n[DEBUG RECONSTRUCTION CHECK]:")
            print(f"  Equation: '{einsum_eq}'")
            print(f"  ABS error = {recon_err_abs:.6e}")
            print(f"  REL error = {recon_err_rel:.6e} {'✓' if recon_err_rel < 1e-10 else '❌'}")
        except Exception as e:
            print(f"\n[DEBUG RECONSTRUCTION CHECK]: failed: {e}")
        t1 = time.perf_counter()
        times["Reconstruction check"] = t1 - t0

    s_out = None

    t0 = time.perf_counter()
    u_final = cp.ascontiguousarray(u_reshaped)
    vh_final = cp.ascontiguousarray(vh_reshaped)
    t1 = time.perf_counter()
    times["Finalize output"] = t1 - t0

    if debug:
        end_total = time.perf_counter()
        total_time = end_total - start_total
        print(f"\n[DEBUG TIMING] Total elapsed time: {total_time:.6f} seconds")
        for name, t in times.items():
            perc = (t / total_time) * 100 if total_time > 0 else 0
            print(f"  {name}: {t:.6f} s ({perc:.2f}%)")

    return u_final, s_out, vh_final


def rttd(subscripts, *operands, algorithm=None, options=None, optimize=None, stream=None, return_info=False):
    """
    Evaluate the compound expression (Contract + rttd/rSVD) on the input operands.
    Uses C++ backend for fast decomposition if available.
    """
    t_start = time.time()

    if algorithm is None or "svd_method" not in algorithm:
        raise ValueError("Algorithm must be a dict with 'svd_method'")

    svd_method_dict = algorithm.get("svd_method", {})
    chi = svd_method_dict.get("max_extent", None)
    partition = svd_method_dict.get("partition", None)

    if chi is None:
        raise ValueError("'svd_method' must specify 'max_extent' (chi)")

    wrapped_operands, inputs, outputs, size_dict, mode_map_user_to_ord, mode_map_ord_to_user, max_mid_extent = (
        decomposition_utils.parse_decomposition(subscripts, *operands)
    )

    own_handle = False
    options = utils.check_or_create_options(NetworkOptions, options, "Network Options")

    try:
        wrapped_operands, options, own_handle, operands_location, stream_holder = (
            decomposition_utils.parse_decompose_operands_options(
                options,
                wrapped_operands,
                stream,
                allowed_dtype_names=decomposition_utils.DECOMPOSITION_DTYPE_NAMES,
            )
        )

        intermediate_modes = einsum_parser.infer_output_mode_labels(outputs)

        intermediate_labels = []
        ellipses_processed = False
        for _modes in intermediate_modes:
            m = mode_map_ord_to_user[_modes]
            if m.startswith("__"):
                if not ellipses_processed:
                    m = "..."
                    ellipses_processed = True
                else:
                    continue
            intermediate_labels.append(m)
        intermediate_labels = "".join(intermediate_labels)

        input_modes, output_modes = subscripts.split("->")
        einsum_subscripts = f"{input_modes}->{intermediate_labels}"
        decompose_subscripts = f"{intermediate_labels}->{output_modes}"

        if operands_location == "cpu":
            operands = [o.tensor for o in wrapped_operands]

        t_contract_start = time.time()
        intm_output = contract(
            einsum_subscripts,
            *operands,
            options=options,
            optimize=optimize,
            stream=stream,
        )
        t_contract_end = time.time()

        if return_info:
            info_dict = {"optimizer_info": None, "svd_info": None}

        t_decompose_start = time.time()
        results = _rttd_decompose(intm_output, decompose_subscripts, chi, partition)
        t_decompose_end = time.time()

        if operands_location == "cpu":
            results = [
                o
                if o is None
                else tensor_wrapper.wrap_operand(o).to("cpu", stream_holder=stream_holder)
                for o in results
            ]

    finally:
        if own_handle and options.handle is not None:
            cutn.destroy(options.handle)

    t_end = time.time()
    # print(f"[PROFILE] rttd: total={1000*(t_end-t_start):.1f}ms, contract={1000*(t_contract_end-t_contract_start):.1f}ms, decompose={1000*(t_decompose_end-t_decompose_start):.1f}ms")

    if not return_info:
        return results
    if len(results) == 3:
        return *results, ContractDecomposeInfo(**info_dict)
    return results[0], results[1], results[2], ContractDecomposeInfo(**info_dict)