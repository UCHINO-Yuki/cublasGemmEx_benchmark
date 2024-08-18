#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <fstream>
#include <cxxabi.h>
#include <chrono>

cudaEvent_t start, stop;

void startTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

float stopTimer() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

template <typename T>
__global__ void set_mat_kernel(std::size_t m, std::size_t n, T *Mat, T value) {
	const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= m*n) {
		return;
	}
	Mat[tid] = value;
}

void test_sq_mat(int rep, std::size_t nmin, std::size_t nmax){
    // setting for cublasGemmEx
    // Change yourself!!
    using IN = std::int8_t;
    using OUT = std::int32_t;
    const auto typeIN = CUDA_R_8I;
    const auto typeOUT = CUDA_R_32I;
    const auto CompMode = CUBLAS_COMPUTE_32I;

    // handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // set scalars
    const OUT alpha = 1;
    const OUT beta = 0;

    // set matrices
    std::size_t sizeAmax = nmax * nmax;
    IN *d_A, *d_B;
    OUT *d_C;
    cudaMalloc(&d_A, sizeAmax * sizeof(IN));
    cudaMalloc(&d_B, sizeAmax * sizeof(IN));
    cudaMalloc(&d_C, sizeAmax * sizeof(OUT));
	dim3 threads = 256;	// <= 1024
	dim3 grid = (sizeAmax + 256 - 1) / 256;
	set_mat_kernel<IN> <<< grid, threads >>> (nmax, nmax, d_A, 1);
	set_mat_kernel<IN> <<< grid, threads >>> (nmax, nmax, d_B, 1);
	set_mat_kernel<OUT> <<< grid, threads >>> (nmax, nmax, d_C, 0);

    // evaluate performance
    for(std::size_t n = nmin; n <= nmax; n <<= 1){
        std::ofstream file("gemmEx_test_result.csv", std::ios::app);

        // warm up
        cublasStatus_t stat;
        for (int j = 0; j < 10; j++) {
            stat = cublasGemmEx(handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                n, n, n,
                                &alpha,
                                d_A, typeIN, n,
                                d_B, typeIN, n,
                                &beta,
                                d_C, typeOUT, n,
                                CompMode, CUBLAS_GEMM_DEFAULT);
        }
        if(stat != CUBLAS_STATUS_SUCCESS){
            std::cout << "cublasGemmEx failed" << std::endl;
        }

        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            std::printf("Kernel launch error: %s\n", cudaGetErrorString(err));
            return ;
        }
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::printf("Kernel execution error: %s\n", cudaGetErrorString(err));
            return ;
        }
        
        float ms = 0.0;
        for(int j = 0; j < rep; j++){
            startTimer();

            stat = cublasGemmEx(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_A, typeIN, n,
                d_B, typeIN, n,
                &beta,
                d_C, typeOUT, n,
                CompMode, CUBLAS_GEMM_DEFAULT);
            
            ms += stopTimer();
             
            if(stat != CUBLAS_STATUS_SUCCESS){
                std::cout << "cublasGemmEx failed" << std::endl;
            }

            cudaError_t err = cudaPeekAtLastError();
            if (err != cudaSuccess) {
                std::printf("Kernel launch error: %s\n", cudaGetErrorString(err));
                return ;
            }
     
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::printf("Kernel execution error: %s\n", cudaGetErrorString(err));
                return ;
            }
        }
        cudaDeviceSynchronize();

        ms /= rep;
        int status;
        std::printf("cublasGemmEx input: %s, output: %s, %d x %d x %d takes %e ms, TOPS is %e\n", 
            abi::__cxa_demangle(typeid(IN).name(),0,0,&status),
            abi::__cxa_demangle(typeid(OUT).name(),0,0,&status),
            (int)n, (int)n, (int)n, 
            ms, 2.0*n*n*n/ms/1e9);

        file << n << "," << ms << "\n";
        file.close();
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaDeviceReset();
}

int main(int argc, char *argv[]){
    // # of iteration
    int rep = 50;
    if (argc > 1) {
        rep = atoi(argv[1]);
    }
    
    // min of log2(n)
    std::size_t nmin = (std::size_t)std::pow(2,8);
    if (argc > 2) {
        nmin = (std::size_t)std::pow(2,atoi(argv[2]));
    }
    
    // max of log2(n)
    std::size_t nmax = (std::size_t)std::pow(2,16);
    if (argc > 3) {
        nmax = (std::size_t)std::pow(2,atoi(argv[3]));
    }

    // execution
    test_sq_mat(rep,nmin,nmax);

    return 0;
}