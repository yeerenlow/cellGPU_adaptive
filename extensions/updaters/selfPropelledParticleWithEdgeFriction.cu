#include "std_include.h"
#include "curand_kernel.h"
#include "indexer.h"
#include "selfPropelledParticleWithEdgeFriction.cuh"
//#include <chrono>  // added 07-01-2025
//#include <iostream>  // added 07-01-2025
#include <cassert>  // added 07-02-2025

__global__ void init_old_neighbors_kernel(int* old_nn, int* old_n, int N)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int MAX_NEIGHS = 16;

    old_nn[idx] = 0;
    for (int k = 0; k < MAX_NEIGHS; ++k)
        {
        old_n[MAX_NEIGHS*idx+k] = -1;
        }
    }

__global__ void checkNeighborChange_kernel(
    int* old_nn,
    int* old_n,
    const int* __restrict__ new_nn,
    const int* __restrict__ new_n,
    Index2D n_idx,
    int N,
    int* changed)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const int MAX_NEIGHS = 16;

    assert(new_nn[idx] <= MAX_NEIGHS);

    if (old_nn[idx] != new_nn[idx])
        {
        atomicExch(changed, 1);
        old_nn[idx] = new_nn[idx];
        }

    int nn = new_nn[idx];
    for (int k = 0; k < nn; ++k)
        {
        if (old_n[n_idx(k,idx)] != new_n[n_idx(k,idx)])
            {
            atomicExch(changed, 1);
            old_n[n_idx(k,idx)] = new_n[n_idx(k,idx)];
            }
        }
    }

__global__ void calculateForces_kernel(
    const double2* __restrict__ forces,
    const double2* __restrict__ motility,
    const double* __restrict__ cellDirectors,
    double* totalf_flat,
    int N)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    // Calculate the total force vector
    double v0 = motility[idx].x;
    double cosValue, sinValue;
    sincos(cellDirectors[idx],&sinValue,&cosValue);
    totalf_flat[2*idx] = v0 * cosValue + forces[idx].x;
    totalf_flat[2*idx+1] = v0 * sinValue + forces[idx].y;
    }

__global__ void fillRowSize_kernel(int* row_sizes, const int* __restrict__ d_nn, int N)
    {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2*N) return;
    int cellidx = i / 2;  // Determine the cell index
    row_sizes[i] = 2*(d_nn[cellidx] + 1);
    }

__global__ void buildFrictionMatrixCSR_kernel(
    const double2* __restrict__ cellPositions,
    const int* __restrict__ neighborNum,
    const int* __restrict__ neighbors,
    Index2D n_idx,
    periodicBoundaries Box,
    int N,
    double gamma_sub,
    double gamma_rel,
    int* row_ptr,
    int* col_idx,
    double* values)
    {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double Pthreshold = THRESHOLD;
    int row_start[2] = {row_ptr[2*i], row_ptr[2*i+1]};
    int nn = neighborNum[i];
    int offset = 0;
    double2 vec;
    double normsq, temp;
    double aux[3] = {0., 0., 0.};
    for (int k = 0; k < nn; ++k)
        {
        int j = neighbors[n_idx(k,i)];
        Box.minDist(cellPositions[i], cellPositions[j], vec);
        normsq = vec.x*vec.x + vec.y*vec.y;
        if (normsq < Pthreshold) normsq = Pthreshold;
        temp = vec.y*vec.y/normsq;
        aux[0] += temp;
        col_idx[row_start[0] + 2*offset] = 2*j;
        values[row_start[0] + 2*offset] = -gamma_rel*temp;
        temp = -vec.x*vec.y/normsq;
        aux[1] += temp;
        col_idx[row_start[0] + 2*offset + 1] = 2*j + 1;
        values[row_start[0] + 2*offset + 1] = -gamma_rel*temp;
        col_idx[row_start[1] + 2*offset] = 2*j;
        values[row_start[1] + 2*offset] = -gamma_rel*temp;
        temp = vec.x*vec.x/normsq;
        aux[2] += temp;
        col_idx[row_start[1] + 2*offset + 1] = 2*j + 1;
        values[row_start[1] + 2*offset + 1] = -gamma_rel*temp;
        ++offset;
        }
    col_idx[row_start[0] + 2*offset] = 2*i;
    values[row_start[0] + 2*offset] = gamma_sub + gamma_rel*aux[0];
    col_idx[row_start[0] + 2*offset + 1] = 2*i + 1;
    values[row_start[0] + 2*offset + 1] = gamma_rel*aux[1];
    col_idx[row_start[1] + 2*offset] = 2*i;
    values[row_start[1] + 2*offset] = gamma_rel*aux[1];
    col_idx[row_start[1] + 2*offset + 1] = 2*i + 1;
    values[row_start[1] + 2*offset + 1] = gamma_sub + gamma_rel*aux[2];
    }

__global__ void spp_friction_eom_integration_kernel(
                                        double2* displacements,
                                        const double2* __restrict__ motility,
                                        double* cellDirectors,
                                        double2* velocities,
                                        const double* __restrict__ velocity_flat,
                                        curandState* RNGs,
                                        int N,
                                        double deltaT,
                                        int Timestep)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    velocities[idx].x = velocity_flat[2*idx];
    velocities[idx].y = velocity_flat[2*idx+1];
    // update displacements
    displacements[idx].x = deltaT*velocities[idx].x;
    displacements[idx].y = deltaT*velocities[idx].y;

    //next, get an appropriate random angle displacement
    curandState_t randState;
    randState = RNGs[idx];
    double Dr = motility[idx].y;
    double angleDiff = cur_norm(&randState)*sqrt(2.0*deltaT*Dr);
    RNGs[idx] = randState;
    //update director
    double tempTheta = cellDirectors[idx];
    // ensure that angle is between -pi and pi
    tempTheta += angleDiff;
    if (tempTheta > M_PI)
        tempTheta -= 2*M_PI;
    else if (tempTheta <= -M_PI)
        tempTheta += 2*M_PI;
    cellDirectors[idx] = tempTheta;
    return;
    }

bool gpu_init_old_neighbors(int* old_nn, int* old_n, int N)
    {
    int blockSize = 256;
    int nBlocks = (N + blockSize - 1) / blockSize;
    init_old_neighbors_kernel<<<nBlocks,blockSize>>>(old_nn, old_n, N);

    // added 07-01-2025
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        return false;
        }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        {
        std::cerr << "Device sync failed: " << cudaGetErrorString(err) << "\n";
        return false;
        }

    return true;
    }

int gpu_computeRowPtr(
    const int* d_nn,
    int N,
    int* d_row_ptr,
    int* d_row_sizes)
    {
    // Calculate the row sizes
    int nrows = 2 * N;
    int blockSize = 256;
    int nBlocks = (nrows + blockSize - 1) / blockSize;
    fillRowSize_kernel<<<nBlocks, blockSize>>>(d_row_sizes, d_nn, N);
    // Do inclusive scan using Thrust to get row_ptr
    thrust::inclusive_scan(thrust::device, d_row_sizes, d_row_sizes + nrows, d_row_ptr+1);
    int nnz = thrust::reduce(thrust::device, d_row_sizes, d_row_sizes + nrows, 0, thrust::plus<int>());
    return nnz;
    }

bool gpu_spp_friction_eom_integration(
                            int* new_nn,
                            int* new_n,
                            int* old_nn,
                            int* old_n,
                            int nnz,
                            const double2* cellPositions,
                            const double2* forces,
                            double2* velocities,
                            double* velocity_flat,
                            double* totalf_flat,
                            double2* displacements,
                            const double2 *motility,
                            double* cellDirectors,
                            int* d_row_ptr,
                            int* d_col_idx,
                            double* d_values,
                            int* d_row_sizes,
                            curandState* RNGs,
                            int N,
                            Index2D n_idx,
                            periodicBoundaries Box,
                            double deltaT,
                            int Timestep,
                            double gamma_sub,
                            double gamma_rel,
                            cudssHandle_t handle,
                            cudssConfig_t config,
                            cudssData_t data,
                            cudssMatrix_t A,
                            cudssMatrix_t b,
                            cudssMatrix_t x,
                            int* d_neigh_change)
    {
    int blockSize = 128;
    int nBlocks = (N + blockSize - 1) / blockSize;
    //double elapsed_ms = 0.0;
    // check neighbor change
    cudaMemset(d_neigh_change, 0, sizeof(int));
    //auto start = std::chrono::high_resolution_clock::now();
    checkNeighborChange_kernel<<<nBlocks,blockSize>>>(old_nn,old_n,new_nn,new_n,n_idx,N,d_neigh_change);
    //auto end = std::chrono::high_resolution_clock::now();
    //elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    //cout << "checkNeighborChange_kernel took " << elapsed_ms << " ms" << endl;
    int h_neigh_change = 0;
    cudaMemcpy(&h_neigh_change, d_neigh_change, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_neigh_change != 0)
        {
        //start = std::chrono::high_resolution_clock::now();
        int nnz_check = gpu_computeRowPtr(new_nn, N, d_row_ptr, d_row_sizes);
        //end = std::chrono::high_resolution_clock::now();
        //elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        //cout << "gpu_computeRowPtr took " << elapsed_ms << " ms" << endl;
        if (nnz_check != nnz)
            {
            cout << "nnz changes from " << nnz << " to " << nnz_check << endl;
            nnz = nnz_check;
            //resize the related things
            cudaMalloc(&d_col_idx, nnz * sizeof(int));
            cudaMalloc(&d_values, nnz * sizeof(double));
            cudssMatrixCreateCsr(
                        &A,
                        /*rows=*/2*N,
                        /*cols=*/2*N,
                        nnz,
                        d_row_ptr,
                        NULL,
                        d_col_idx,
                        d_values,
                        CUDA_R_32I, // index type
                        CUDA_R_64F, // value type
                        CUDSS_MTYPE_SPD, // Matrix type symmetric positive definite
                        CUDSS_MVIEW_FULL, // Matirx view type
                        CUDSS_BASE_ZERO);
            }
        }
    //start = std::chrono::high_resolution_clock::now();
    buildFrictionMatrixCSR_kernel<<<nBlocks,blockSize>>>(
        cellPositions,
        new_nn,
        new_n,
        n_idx,
        Box,
        N,
        gamma_sub,
        gamma_rel,
        d_row_ptr,
        d_col_idx,
        d_values);
    //end = std::chrono::high_resolution_clock::now();
    //elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    //cout << "buildFrictionMatrixCSR_kernel took " << elapsed_ms << " ms" << endl;
    if (h_neigh_change != 0)
        {
        // Symbolic analysis (pattern only)
        //start = std::chrono::high_resolution_clock::now();
        cudssExecute(handle,
            CUDSS_PHASE_ANALYSIS,
            config,
            data,
            A, x, b);
        //end = std::chrono::high_resolution_clock::now();
        //elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        //cout << "symbolic analysis took " << elapsed_ms << " ms" << endl;
        }
    // calculate the total forces vector, which automatically update b
    //start = std::chrono::high_resolution_clock::now();
    calculateForces_kernel<<<nBlocks,blockSize>>>(forces, motility, cellDirectors, totalf_flat, N);
    //end = std::chrono::high_resolution_clock::now();
    //elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    //cout << "calculateForces_kernel took " << elapsed_ms << " ms" << endl;
    // Numeric factorization
    //start = std::chrono::high_resolution_clock::now();
    cudssExecute(handle,
            CUDSS_PHASE_FACTORIZATION,
            config,
            data,
            A, x, b);
    //end = std::chrono::high_resolution_clock::now();
    //elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    //cout << "factorization took " << elapsed_ms << " ms" << endl;
    // Solve
    //start = std::chrono::high_resolution_clock::now();
    cudssExecute(handle,
            CUDSS_PHASE_SOLVE,
            config,
            data,
            A, x, b);
    //end = std::chrono::high_resolution_clock::now();
    //elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    //cout << "solve took " << elapsed_ms << " ms" << endl;
    // integration and update
    //start = std::chrono::high_resolution_clock::now();
    spp_friction_eom_integration_kernel<<<nBlocks,blockSize>>>(
        displacements,
        motility,
        cellDirectors,
        velocities,
        velocity_flat,
        RNGs,
        N,
        deltaT,
        Timestep);
    //end = std::chrono::high_resolution_clock::now();
    //elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    //cout << "spp_friction_eom_integration_kernel took " << elapsed_ms << " ms" << endl;
    return true;
    }
