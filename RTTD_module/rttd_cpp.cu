#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>

namespace py = pybind11;

#define CHECK_CUDA(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e)); \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) \
        throw std::runtime_error("cuBLAS error: " + std::to_string(s)); \
} while(0)

#define CHECK_CUSOLVER(call) do { \
    cusolverStatus_t s = call; \
    if (s != CUSOLVER_STATUS_SUCCESS) \
        throw std::runtime_error("cuSOLVER error: " + std::to_string(s)); \
} while(0)

#define CHECK_CURAND(call) do { \
    curandStatus_t s = call; \
    if (s != CURAND_STATUS_SUCCESS) \
        throw std::runtime_error("cuRAND error: " + std::to_string(s)); \
} while(0)

// OPTIMIZED: Coalesced transpose kernel with tiling
template<int TILE_DIM>
__global__ void transpose_kernel(const double* __restrict__ in, double* __restrict__ out, int M, int N) {
    __shared__ double tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    // Read from global memory (coalesced)
    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }
    
    __syncthreads();
    
    // Write to global memory (coalesced after transpose)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    if (x < M && y < N) {
        out[y * M + x] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose_rowmajor_to_colmajor(const double* d_in, double* d_out, int M, int N) {
    const int TILE = 32;
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    dim3 block(TILE, TILE);
    transpose_kernel<TILE><<<grid, block>>>(d_in, d_out, M, N);
}

double* get_cupy_ptr(py::object arr) {
    return reinterpret_cast<double*>(arr.attr("data").attr("ptr").cast<uintptr_t>());
}

std::vector<size_t> get_shape(py::object arr) {
    return arr.attr("shape").cast<std::vector<size_t>>();
}

py::object make_contiguous_cupy(double* dev_ptr, const std::vector<size_t>& shape) {
    py::module_ cp = py::module_::import("cupy");
    size_t nelem = 1;
    for (auto s : shape) nelem *= s;
    size_t nbytes = nelem * sizeof(double);
    
    // Allocate new CuPy-owned memory
    py::object mem = cp.attr("cuda").attr("Memory")(py::int_(nbytes));
    
    // Get pointer to CuPy memory
    uintptr_t cupy_addr = mem.attr("ptr").cast<uintptr_t>();
    double* cupy_ptr = reinterpret_cast<double*>(cupy_addr);
    
    // Copy data from dev_ptr to CuPy memory
    CHECK_CUDA(cudaMemcpy(cupy_ptr, dev_ptr, nbytes, cudaMemcpyDeviceToDevice));
    
    // Free the original allocation (transfer ownership complete)
    CHECK_CUDA(cudaFree(dev_ptr));
    
    py::object memptr = cp.attr("cuda").attr("MemoryPointer")(mem, 0);
    return cp.attr("ndarray")(py::cast(shape), cp.attr("float64"), memptr);
}

py::object make_cupy_array_from_vec(const std::vector<double>& h_vec) {
    size_t n = h_vec.size();
    double* d_arr;
    CHECK_CUDA(cudaMalloc(&d_arr, n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_arr, h_vec.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    return make_contiguous_cupy(d_arr, std::vector<size_t>{n});
}

py::tuple rttd_cpp(py::object tensor_in, std::string decompose_subscripts, int chi, 
                    std::string partition = "U") {
    const int num_iter = 1;
    const int p = 5; // oversampling parameter
    double one = 1.0, zero = 0.0;
    CHECK_CUDA(cudaSetDevice(0));

    auto shp = get_shape(tensor_in);
    if (shp.size() != 2) {
        throw std::runtime_error("rttd_cpp expects 2D input. Python should pre-flatten.");
    }
    
    size_t M = shp[0];
    size_t N = shp[1];
    // Parse decompose_subscripts to get permutation

    double* d_in_rowmajor = get_cupy_ptr(tensor_in);

    double *d_mat_colmajor;
    CHECK_CUDA(cudaMalloc(&d_mat_colmajor, M * N * sizeof(double)));
    transpose_rowmajor_to_colmajor(d_in_rowmajor, d_mat_colmajor, M, N);

    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH;
    curandGenerator_t curandG;
    CHECK_CUBLAS(cublasCreate(&cublasH));
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
    CHECK_CURAND(curandCreateGenerator(&curandG, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(curandG, 1234ULL));

    // size_t q = std::min(static_cast<size_t>(chi) + 10, std::min(M, N));
    size_t q = std::min(static_cast<size_t>(chi) + p, std::min(M, N));
    size_t r = std::min(static_cast<size_t>(chi), std::min(M, N));

    py::object u_out, s_out, vh_out;

    if (partition == "U") {
        // ===== PARTITION 'U': QR-BASED (LEFT-CANONICAL) =====
        double *d_omega, *d_Y, *d_Z, *d_Q, *d_tau, *d_work = nullptr;
        int *d_info;
        int lwork = 0;

        CHECK_CUDA(cudaMalloc(&d_omega, N * q * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_Y, M * q * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_Z, N * q * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_Q, M * q * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_tau, std::min(M, q) * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

        CHECK_CURAND(curandGenerateNormalDouble(curandG, d_omega, N * q, 0.0, 1.0 / std::sqrt(static_cast<double>(q))));

        // Y = A @ omega
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(M), static_cast<int>(q), static_cast<int>(N),
            &one, d_mat_colmajor, static_cast<int>(M),
            d_omega, static_cast<int>(N),
            &zero, d_Y, static_cast<int>(M)));

        // Power iteration
        for (int iter = 0; iter < num_iter; ++iter) {
            // Z = A^T @ Y
            CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(N), static_cast<int>(q), static_cast<int>(M),
                &one, d_mat_colmajor, static_cast<int>(M), d_Y, static_cast<int>(M),
                &zero, d_Z, static_cast<int>(N)));

            // QR(Z)
            double *d_Q_temp, *d_tau_temp, *d_work_temp = nullptr;
            int *d_info_temp, lwork_qr_z;
            CHECK_CUDA(cudaMalloc(&d_Q_temp, N * q * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&d_tau_temp, std::min(N, q) * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&d_info_temp, sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_Q_temp, d_Z, N * q * sizeof(double), cudaMemcpyDeviceToDevice));

            CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, static_cast<int>(N), static_cast<int>(q), d_Q_temp, static_cast<int>(N), &lwork_qr_z));
            if (lwork_qr_z > 0) CHECK_CUDA(cudaMalloc(&d_work_temp, lwork_qr_z * sizeof(double)));
            CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, static_cast<int>(N), static_cast<int>(q), d_Q_temp, static_cast<int>(N), d_tau_temp, d_work_temp, lwork_qr_z, d_info_temp));

            int lwork_orgqr_z;
            CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(cusolverH, static_cast<int>(N), static_cast<int>(q), static_cast<int>(std::min(N, q)), d_Q_temp, static_cast<int>(N), d_tau_temp, &lwork_orgqr_z));
            if (lwork_orgqr_z > lwork_qr_z) {
                if (d_work_temp) CHECK_CUDA(cudaFree(d_work_temp));
                CHECK_CUDA(cudaMalloc(&d_work_temp, lwork_orgqr_z * sizeof(double)));
            }
            CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, static_cast<int>(N), static_cast<int>(q), static_cast<int>(std::min(N, q)), d_Q_temp, static_cast<int>(N), d_tau_temp, d_work_temp, lwork_orgqr_z, d_info_temp));
            CHECK_CUDA(cudaMemcpy(d_Z, d_Q_temp, N * q * sizeof(double), cudaMemcpyDeviceToDevice));
            if (d_work_temp) CHECK_CUDA(cudaFree(d_work_temp));
            CHECK_CUDA(cudaFree(d_Q_temp)); CHECK_CUDA(cudaFree(d_tau_temp)); CHECK_CUDA(cudaFree(d_info_temp));

            // Y = A @ Z
            CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(M), static_cast<int>(q), static_cast<int>(N),
                &one, d_mat_colmajor, static_cast<int>(M), d_Z, static_cast<int>(N),
                &zero, d_Y, static_cast<int>(M)));

            // QR(Y)
            double *d_Q_temp2, *d_tau_temp2, *d_work_temp2 = nullptr;
            int *d_info_temp2, lwork_qr_y;
            CHECK_CUDA(cudaMalloc(&d_Q_temp2, M * q * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&d_tau_temp2, std::min(M, q) * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&d_info_temp2, sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_Q_temp2, d_Y, M * q * sizeof(double), cudaMemcpyDeviceToDevice));

            CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, static_cast<int>(M), static_cast<int>(q), d_Q_temp2, static_cast<int>(M), &lwork_qr_y));
            if (lwork_qr_y > 0) CHECK_CUDA(cudaMalloc(&d_work_temp2, lwork_qr_y * sizeof(double)));
            CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, static_cast<int>(M), static_cast<int>(q), d_Q_temp2, static_cast<int>(M), d_tau_temp2, d_work_temp2, lwork_qr_y, d_info_temp2));

            int lwork_orgqr_y;
            CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(cusolverH, static_cast<int>(M), static_cast<int>(q), static_cast<int>(std::min(M, q)), d_Q_temp2, static_cast<int>(M), d_tau_temp2, &lwork_orgqr_y));
            if (lwork_orgqr_y > lwork_qr_y) {
                if (d_work_temp2) CHECK_CUDA(cudaFree(d_work_temp2));
                CHECK_CUDA(cudaMalloc(&d_work_temp2, lwork_orgqr_y * sizeof(double)));
            }
            CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, static_cast<int>(M), static_cast<int>(q), static_cast<int>(std::min(M, q)), d_Q_temp2, static_cast<int>(M), d_tau_temp2, d_work_temp2, lwork_orgqr_y, d_info_temp2));
            CHECK_CUDA(cudaMemcpy(d_Y, d_Q_temp2, M * q * sizeof(double), cudaMemcpyDeviceToDevice));
            if (d_work_temp2) CHECK_CUDA(cudaFree(d_work_temp2));
            CHECK_CUDA(cudaFree(d_Q_temp2)); CHECK_CUDA(cudaFree(d_tau_temp2)); CHECK_CUDA(cudaFree(d_info_temp2));
        }

        // Final QR on Y
        CHECK_CUDA(cudaMemcpy(d_Q, d_Y, M * q * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, static_cast<int>(M), static_cast<int>(q), d_Q, static_cast<int>(M), &lwork));
        if (lwork > 0) CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(double)));
        CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, static_cast<int>(M), static_cast<int>(q), d_Q, static_cast<int>(M), d_tau, d_work, lwork, d_info));

        int lwork_orgqr;
        CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(cusolverH, static_cast<int>(M), static_cast<int>(q), static_cast<int>(std::min(M, q)), d_Q, static_cast<int>(M), d_tau, &lwork_orgqr));
        if (lwork_orgqr > lwork) {
            if (d_work) CHECK_CUDA(cudaFree(d_work));
            CHECK_CUDA(cudaMalloc(&d_work, lwork_orgqr * sizeof(double)));
        }
        CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, static_cast<int>(M), static_cast<int>(q), static_cast<int>(std::min(M, q)), d_Q, static_cast<int>(M), d_tau, d_work, lwork_orgqr, d_info));

        // Truncate Q to r columns
        double *d_U_colmajor;
        CHECK_CUDA(cudaMalloc(&d_U_colmajor, M * r * sizeof(double)));
        for (size_t k = 0; k < r; ++k) {
            CHECK_CUDA(cudaMemcpy(d_U_colmajor + k * M, d_Q + k * M, M * sizeof(double), cudaMemcpyDeviceToDevice));
        }

        // R = Q^T @ A (r x N, upper triangular approx)
        // R = Q^T @ A (r x N)
        double *d_R;
        CHECK_CUDA(cudaMalloc(&d_R, r * N * sizeof(double)));
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(r), static_cast<int>(N), static_cast<int>(M),
            &one, d_U_colmajor, static_cast<int>(M),
            d_mat_colmajor, static_cast<int>(M),
            &zero, d_R, static_cast<int>(r)));

        // QR on R^T to get orthonormal Vh and extract S from R diagonal
        // R^T is N x r, QR gives: R^T = Vh^T @ R2 where Vh^T is N x r orthonormal, R2 is r x r upper triangular
        // Then R = R2^T @ Vh, and since R = (Q^T A), we have A = Q @ R2^T @ Vh

        // Transpose R to R^T (N x r)
        double *d_R_T;
        CHECK_CUDA(cudaMalloc(&d_R_T, N * r * sizeof(double)));
        CHECK_CUBLAS(cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(N), static_cast<int>(r),
            &one, d_R, static_cast<int>(r),
            &zero, d_R, static_cast<int>(N),
            d_R_T, static_cast<int>(N)));

        // QR on R^T (N x r)
        double *d_Vh_T, *d_tau_rt, *d_work_rt = nullptr;
        int *d_info_rt, lwork_rt;
        CHECK_CUDA(cudaMalloc(&d_Vh_T, N * r * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_tau_rt, r * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_info_rt, sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_Vh_T, d_R_T, N * r * sizeof(double), cudaMemcpyDeviceToDevice));

        CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, static_cast<int>(N), static_cast<int>(r), d_Vh_T, static_cast<int>(N), &lwork_rt));
        if (lwork_rt > 0) CHECK_CUDA(cudaMalloc(&d_work_rt, lwork_rt * sizeof(double)));
        CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, static_cast<int>(N), static_cast<int>(r), d_Vh_T, static_cast<int>(N), d_tau_rt, d_work_rt, lwork_rt, d_info_rt));

        // Extract R2 (upper triangular r x r) from d_Vh_T before orgqr
        double *d_R2;
        CHECK_CUDA(cudaMalloc(&d_R2, r * r * sizeof(double)));
        CHECK_CUDA(cudaMemset(d_R2, 0, r * r * sizeof(double)));
        for (size_t j = 0; j < r; ++j) {
            // Copy column j of upper triangular part (rows 0 to j)
            CHECK_CUDA(cudaMemcpy(d_R2 + j * r, d_Vh_T + j * N, (j + 1) * sizeof(double), cudaMemcpyDeviceToDevice));
        }

        // Extract diagonal of R2 as singular values
        std::vector<double> h_S(r);
        for (size_t i = 0; i < r; ++i) {
            CHECK_CUDA(cudaMemcpy(&h_S[i], d_R2 + i * r + i, sizeof(double), cudaMemcpyDeviceToHost));
            h_S[i] = std::abs(h_S[i]);  // Take absolute value
        }

        // Generate orthonormal Q from QR
        int lwork_orgqr_rt;
        CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(cusolverH, static_cast<int>(N), static_cast<int>(r), static_cast<int>(r), d_Vh_T, static_cast<int>(N), d_tau_rt, &lwork_orgqr_rt));
        if (lwork_orgqr_rt > lwork_rt) {
            if (d_work_rt) CHECK_CUDA(cudaFree(d_work_rt));
            CHECK_CUDA(cudaMalloc(&d_work_rt, lwork_orgqr_rt * sizeof(double)));
        }
        CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, static_cast<int>(N), static_cast<int>(r), static_cast<int>(r), d_Vh_T, static_cast<int>(N), d_tau_rt, d_work_rt, lwork_orgqr_rt, d_info_rt));

        // Vh_T is now N x r orthonormal columns, transpose to get Vh (r x N)
        double *d_Vh_final;
        CHECK_CUDA(cudaMalloc(&d_Vh_final, r * N * sizeof(double)));
        CHECK_CUBLAS(cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(r), static_cast<int>(N),
            &one, d_Vh_T, static_cast<int>(N),
            &zero, d_Vh_T, static_cast<int>(r),
            d_Vh_final, static_cast<int>(r)));

        // Now we have: R^T = Vh^T @ R2, so R = R2^T @ Vh
        // A ≈ Q @ R = Q @ R2^T @ Vh, but we want A = U_final @ Vh with S absorbed in U_final
        // So U_final = Q @ R2^T with columns scaled by |diag(R2)|

        // Compute U_final = Q @ R2^T
        double *d_R2_T;
        CHECK_CUDA(cudaMalloc(&d_R2_T, r * r * sizeof(double)));
        CHECK_CUBLAS(cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(r), static_cast<int>(r),
            &one, d_R2, static_cast<int>(r),
            &zero, d_R2, static_cast<int>(r),
            d_R2_T, static_cast<int>(r)));

        double *d_U_final;
        CHECK_CUDA(cudaMalloc(&d_U_final, M * r * sizeof(double)));
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(M), static_cast<int>(r), static_cast<int>(r),
            &one, d_U_colmajor, static_cast<int>(M),  // Q (M x r)
            d_R2_T, static_cast<int>(r),               // R2^T (r x r)
            &zero, d_U_final, static_cast<int>(M)));   // U_final (M x r)

        // The diagonal of R2 gives us the scaling, but it's already in U_final
        // We need to normalize U_final columns by their norms and absorb those norms back
        // Actually, since R2 is upper triangular from QR, diag(R2) are the "singular values" already
        // U_final already has the right scaling - no additional absorption needed since we're using R2^T

        // Actually wait - let me reconsider. The issue is:
        // A = Q @ R = Q @ R2^T @ Vh
        // If we want U_abs @ Vh = A where U_abs has S absorbed, then:
        // U_abs should equal Q @ R2^T, which already has the scaling
        // So U_final is already correct! But we should verify the signs...

        // Adjust signs if R2 diagonal is negative (maintain positive singular values)
        for (size_t i = 0; i < r; ++i) {
            if (h_S[i] < 0) {  // This shouldn't happen with abs(), but just in case
                double neg = -1.0;
                CHECK_CUBLAS(cublasDscal(cublasH, static_cast<int>(M), &neg, d_U_final + i * M, 1));
                h_S[i] = -h_S[i];
            }
        }

        // Convert to row-major
        double *d_U_rowmajor, *d_Vh_rowmajor;
        CHECK_CUDA(cudaMalloc(&d_U_rowmajor, M * r * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_Vh_rowmajor, r * N * sizeof(double)));
        transpose_rowmajor_to_colmajor(d_U_final, d_U_rowmajor, r, M);
        transpose_rowmajor_to_colmajor(d_Vh_final, d_Vh_rowmajor, N, r);

        CHECK_CUDA(cudaDeviceSynchronize());

        u_out = make_contiguous_cupy(d_U_rowmajor, std::vector<size_t>{M, r});
        s_out = make_cupy_array_from_vec(h_S);
        vh_out = make_contiguous_cupy(d_Vh_rowmajor, std::vector<size_t>{r, N});

        if (d_work) CHECK_CUDA(cudaFree(d_work));
        if (d_work_rt) CHECK_CUDA(cudaFree(d_work_rt));
        CHECK_CUDA(cudaFree(d_omega)); CHECK_CUDA(cudaFree(d_Y)); CHECK_CUDA(cudaFree(d_Z));
        CHECK_CUDA(cudaFree(d_Q)); CHECK_CUDA(cudaFree(d_tau)); CHECK_CUDA(cudaFree(d_info));
        CHECK_CUDA(cudaFree(d_U_colmajor)); CHECK_CUDA(cudaFree(d_R)); CHECK_CUDA(cudaFree(d_R_T));
        CHECK_CUDA(cudaFree(d_Vh_T)); CHECK_CUDA(cudaFree(d_tau_rt)); CHECK_CUDA(cudaFree(d_info_rt));
        CHECK_CUDA(cudaFree(d_R2)); CHECK_CUDA(cudaFree(d_R2_T)); CHECK_CUDA(cudaFree(d_Vh_final));
        CHECK_CUDA(cudaFree(d_U_final));

    } else if (partition == "V") {
        // ===== PARTITION 'V': RQ-BASED (RIGHT-CANONICAL) =====
        // Implement as QR(A^T)^T to get RQ(A)
        
        // Transpose A to A^T (N x M)
        double *d_A_T;
        CHECK_CUDA(cudaMalloc(&d_A_T, N * M * sizeof(double)));
        // Use cublasDgeam for efficient transpose: A^T = 1.0*A^T + 0.0*A
        CHECK_CUBLAS(cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(N), static_cast<int>(M),
            &one, d_mat_colmajor, static_cast<int>(M),
            &zero, d_mat_colmajor, static_cast<int>(N),
            d_A_T, static_cast<int>(N)));

        // Now do randomized QR on A^T (N x M)
        double *d_omega, *d_Y, *d_Z, *d_Q, *d_tau, *d_work = nullptr;
        int *d_info;
        int lwork = 0;

        CHECK_CUDA(cudaMalloc(&d_omega, M * q * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_Y, N * q * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_Z, M * q * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_Q, N * q * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_tau, std::min(N, q) * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

        CHECK_CURAND(curandGenerateNormalDouble(curandG, d_omega, M * q, 0.0, 1.0 / std::sqrt(static_cast<double>(q))));

        // Y = A^T @ omega (N x q)
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(N), static_cast<int>(q), static_cast<int>(M),
            &one, d_A_T, static_cast<int>(N),
            d_omega, static_cast<int>(M),
            &zero, d_Y, static_cast<int>(N)));

        // Power iteration (mirror of 'U' case with A^T)
        for (int iter = 0; iter < num_iter; ++iter) {
            // Z = A @ Y (use original A, transpose operation)
            CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(M), static_cast<int>(q), static_cast<int>(N),
                &one, d_A_T, static_cast<int>(N), d_Y, static_cast<int>(N),
                &zero, d_Z, static_cast<int>(M)));

            // QR(Z) (M x q)
            double *d_Q_temp, *d_tau_temp, *d_work_temp = nullptr;
            int *d_info_temp, lwork_qr_z;
            CHECK_CUDA(cudaMalloc(&d_Q_temp, M * q * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&d_tau_temp, std::min(M, q) * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&d_info_temp, sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_Q_temp, d_Z, M * q * sizeof(double), cudaMemcpyDeviceToDevice));

            CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, static_cast<int>(M), static_cast<int>(q), d_Q_temp, static_cast<int>(M), &lwork_qr_z));
            if (lwork_qr_z > 0) CHECK_CUDA(cudaMalloc(&d_work_temp, lwork_qr_z * sizeof(double)));
            CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, static_cast<int>(M), static_cast<int>(q), d_Q_temp, static_cast<int>(M), d_tau_temp, d_work_temp, lwork_qr_z, d_info_temp));

            int lwork_orgqr_z;
            CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(cusolverH, static_cast<int>(M), static_cast<int>(q), static_cast<int>(std::min(M, q)), d_Q_temp, static_cast<int>(M), d_tau_temp, &lwork_orgqr_z));
            if (lwork_orgqr_z > lwork_qr_z) {
                if (d_work_temp) CHECK_CUDA(cudaFree(d_work_temp));
                CHECK_CUDA(cudaMalloc(&d_work_temp, lwork_orgqr_z * sizeof(double)));
            }
            CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, static_cast<int>(M), static_cast<int>(q), static_cast<int>(std::min(M, q)), d_Q_temp, static_cast<int>(M), d_tau_temp, d_work_temp, lwork_orgqr_z, d_info_temp));
            CHECK_CUDA(cudaMemcpy(d_Z, d_Q_temp, M * q * sizeof(double), cudaMemcpyDeviceToDevice));
            if (d_work_temp) CHECK_CUDA(cudaFree(d_work_temp));
            CHECK_CUDA(cudaFree(d_Q_temp)); CHECK_CUDA(cudaFree(d_tau_temp)); CHECK_CUDA(cudaFree(d_info_temp));

            // Y = A^T @ Z (N x q)
            CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(N), static_cast<int>(q), static_cast<int>(M),
                &one, d_A_T, static_cast<int>(N), d_Z, static_cast<int>(M),
                &zero, d_Y, static_cast<int>(N)));

            // QR(Y) (N x q)
            double *d_Q_temp2, *d_tau_temp2, *d_work_temp2 = nullptr;
            int *d_info_temp2, lwork_qr_y;
            CHECK_CUDA(cudaMalloc(&d_Q_temp2, N * q * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&d_tau_temp2, std::min(N, q) * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&d_info_temp2, sizeof(int)));
            CHECK_CUDA(cudaMemcpy(d_Q_temp2, d_Y, N * q * sizeof(double), cudaMemcpyDeviceToDevice));

            CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, static_cast<int>(N), static_cast<int>(q), d_Q_temp2, static_cast<int>(N), &lwork_qr_y));
            if (lwork_qr_y > 0) CHECK_CUDA(cudaMalloc(&d_work_temp2, lwork_qr_y * sizeof(double)));
            CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, static_cast<int>(N), static_cast<int>(q), d_Q_temp2, static_cast<int>(N), d_tau_temp2, d_work_temp2, lwork_qr_y, d_info_temp2));

            int lwork_orgqr_y;
            CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(cusolverH, static_cast<int>(N), static_cast<int>(q), static_cast<int>(std::min(N, q)), d_Q_temp2, static_cast<int>(N), d_tau_temp2, &lwork_orgqr_y));
            if (lwork_orgqr_y > lwork_qr_y) {
                if (d_work_temp2) CHECK_CUDA(cudaFree(d_work_temp2));
                CHECK_CUDA(cudaMalloc(&d_work_temp2, lwork_orgqr_y * sizeof(double)));
            }
            CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, static_cast<int>(N), static_cast<int>(q), static_cast<int>(std::min(N, q)), d_Q_temp2, static_cast<int>(N), d_tau_temp2, d_work_temp2, lwork_orgqr_y, d_info_temp2));
            CHECK_CUDA(cudaMemcpy(d_Y, d_Q_temp2, N * q * sizeof(double), cudaMemcpyDeviceToDevice));
            if (d_work_temp2) CHECK_CUDA(cudaFree(d_work_temp2));
            CHECK_CUDA(cudaFree(d_Q_temp2)); CHECK_CUDA(cudaFree(d_tau_temp2)); CHECK_CUDA(cudaFree(d_info_temp2));
        }

        // Final QR on Y (N x q) -> Q is right subspace
        CHECK_CUDA(cudaMemcpy(d_Q, d_Y, N * q * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, static_cast<int>(N), static_cast<int>(q), d_Q, static_cast<int>(N), &lwork));
        if (lwork > 0) CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(double)));
        CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, static_cast<int>(N), static_cast<int>(q), d_Q, static_cast<int>(N), d_tau, d_work, lwork, d_info));

        int lwork_orgqr;
        CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(cusolverH, static_cast<int>(N), static_cast<int>(q), static_cast<int>(std::min(N, q)), d_Q, static_cast<int>(N), d_tau, &lwork_orgqr));
        if (lwork_orgqr > lwork) {
            if (d_work) CHECK_CUDA(cudaFree(d_work));
            CHECK_CUDA(cudaMalloc(&d_work, lwork_orgqr * sizeof(double)));
        }
        CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, static_cast<int>(N), static_cast<int>(q), static_cast<int>(std::min(N, q)), d_Q, static_cast<int>(N), d_tau, d_work, lwork_orgqr, d_info));

        // Truncate Q to r columns (N x r, right subspace)
        double *d_V_colmajor;
        CHECK_CUDA(cudaMalloc(&d_V_colmajor, N * r * sizeof(double)));
        for (size_t k = 0; k < r; ++k) {
            CHECK_CUDA(cudaMemcpy(d_V_colmajor + k * N, d_Q + k * N, N * sizeof(double), cudaMemcpyDeviceToDevice));
        }

        // L = A @ Q (M x r)
        double *d_L;
        CHECK_CUDA(cudaMalloc(&d_L, M * r * sizeof(double)));
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(M), static_cast<int>(r), static_cast<int>(N),
            &one, d_mat_colmajor, static_cast<int>(M),
            d_V_colmajor, static_cast<int>(N),
            &zero, d_L, static_cast<int>(M)));

        // QR on L (M x r) to get orthonormal U and upper triangular L2
        double *d_U_final, *d_tau_l, *d_work_l = nullptr;
        int *d_info_l, lwork_l;
        CHECK_CUDA(cudaMalloc(&d_U_final, M * r * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_tau_l, r * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_info_l, sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_U_final, d_L, M * r * sizeof(double), cudaMemcpyDeviceToDevice));

        CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(cusolverH, static_cast<int>(M), static_cast<int>(r), d_U_final, static_cast<int>(M), &lwork_l));
        if (lwork_l > 0) CHECK_CUDA(cudaMalloc(&d_work_l, lwork_l * sizeof(double)));
        CHECK_CUSOLVER(cusolverDnDgeqrf(cusolverH, static_cast<int>(M), static_cast<int>(r), d_U_final, static_cast<int>(M), d_tau_l, d_work_l, lwork_l, d_info_l));

        // Extract L2 (upper triangular r x r) before orgqr
        double *d_L2;
        CHECK_CUDA(cudaMalloc(&d_L2, r * r * sizeof(double)));
        CHECK_CUDA(cudaMemset(d_L2, 0, r * r * sizeof(double)));
        for (size_t j = 0; j < r; ++j) {
            // Copy column j of upper triangular part (rows 0 to j)
            CHECK_CUDA(cudaMemcpy(d_L2 + j * r, d_U_final + j * M, (j + 1) * sizeof(double), cudaMemcpyDeviceToDevice));
        }

        // Extract diagonal of L2 as singular values
        std::vector<double> h_S(r);
        for (size_t i = 0; i < r; ++i) {
            CHECK_CUDA(cudaMemcpy(&h_S[i], d_L2 + i * r + i, sizeof(double), cudaMemcpyDeviceToHost));
            h_S[i] = std::abs(h_S[i]);
        }

        // Generate orthonormal U from QR
        int lwork_orgqr_l;
        CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(cusolverH, static_cast<int>(M), static_cast<int>(r), static_cast<int>(r), d_U_final, static_cast<int>(M), d_tau_l, &lwork_orgqr_l));
        if (lwork_orgqr_l > lwork_l) {
            if (d_work_l) CHECK_CUDA(cudaFree(d_work_l));
            CHECK_CUDA(cudaMalloc(&d_work_l, lwork_orgqr_l * sizeof(double)));
        }
        CHECK_CUSOLVER(cusolverDnDorgqr(cusolverH, static_cast<int>(M), static_cast<int>(r), static_cast<int>(r), d_U_final, static_cast<int>(M), d_tau_l, d_work_l, lwork_orgqr_l, d_info_l));

        // Now: L = U_final @ L2, and L = A @ Q, so A = U_final @ L2 @ Q^T
        // We want A = U_final @ Vh_final where S is absorbed in Vh_final
        // So Vh_final = L2 @ Q^T

        // Transpose Q (N x r) to Q^T (r x N)
        double *d_Q_T;
        CHECK_CUDA(cudaMalloc(&d_Q_T, r * N * sizeof(double)));
        CHECK_CUBLAS(cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(r), static_cast<int>(N),
            &one, d_V_colmajor, static_cast<int>(N),
            &zero, d_V_colmajor, static_cast<int>(r),
            d_Q_T, static_cast<int>(r)));

        // Vh_final = L2 @ Q^T (r x r) @ (r x N) = r x N
        double *d_Vh_final;
        CHECK_CUDA(cudaMalloc(&d_Vh_final, r * N * sizeof(double)));
        CHECK_CUBLAS(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(r), static_cast<int>(N), static_cast<int>(r),
            &one, d_L2, static_cast<int>(r),     // L2 (r x r)
            d_Q_T, static_cast<int>(r),          // Q^T (r x N)
            &zero, d_Vh_final, static_cast<int>(r)));  // Vh_final (r x N)

        // Vh_final already has the scaling from L2, so S is absorbed

        // Adjust signs if needed
        for (size_t i = 0; i < r; ++i) {
            if (h_S[i] < 0) {
                double neg = -1.0;
                CHECK_CUBLAS(cublasDscal(cublasH, static_cast<int>(N), &neg, d_Vh_final + i, static_cast<int>(r)));
                h_S[i] = -h_S[i];
            }
        }

        // Convert to row-major
        double *d_U_rowmajor, *d_Vh_rowmajor;
        CHECK_CUDA(cudaMalloc(&d_U_rowmajor, M * r * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_Vh_rowmajor, r * N * sizeof(double)));
        transpose_rowmajor_to_colmajor(d_U_final, d_U_rowmajor, r, M);
        transpose_rowmajor_to_colmajor(d_Vh_final, d_Vh_rowmajor, N, r);

        CHECK_CUDA(cudaDeviceSynchronize());

        u_out = make_contiguous_cupy(d_U_rowmajor, std::vector<size_t>{M, r});
        s_out = make_cupy_array_from_vec(h_S);
        vh_out = make_contiguous_cupy(d_Vh_rowmajor, std::vector<size_t>{r, N});

        if (d_work) CHECK_CUDA(cudaFree(d_work));
        if (d_work_l) CHECK_CUDA(cudaFree(d_work_l));
        CHECK_CUDA(cudaFree(d_A_T)); CHECK_CUDA(cudaFree(d_omega)); CHECK_CUDA(cudaFree(d_Y));
        CHECK_CUDA(cudaFree(d_Z)); CHECK_CUDA(cudaFree(d_Q)); CHECK_CUDA(cudaFree(d_tau));
        CHECK_CUDA(cudaFree(d_info)); CHECK_CUDA(cudaFree(d_V_colmajor)); CHECK_CUDA(cudaFree(d_L));
        CHECK_CUDA(cudaFree(d_tau_l)); CHECK_CUDA(cudaFree(d_info_l)); CHECK_CUDA(cudaFree(d_L2));
        CHECK_CUDA(cudaFree(d_Q_T)); CHECK_CUDA(cudaFree(d_Vh_final)); CHECK_CUDA(cudaFree(d_U_final));


    } else {
        throw std::runtime_error("partition='balanced' not implemented in QR/RQ version - use SVD");
    }

    CHECK_CUDA(cudaFree(d_mat_colmajor));
    CHECK_CURAND(curandDestroyGenerator(curandG));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
    CHECK_CUBLAS(cublasDestroy(cublasH));

    return py::make_tuple(u_out, s_out, vh_out);
}



PYBIND11_MODULE(rttd_cpp_module, m) {
    m.doc() = "RTTD C++ CUDA extension module";
    m.def("rttd_cpp", &rttd_cpp, 
          py::arg("tensor_in"), 
          py::arg("decompose_subscripts"), 
          py::arg("chi"), 
          py::arg("partition") = "balanced");
}
