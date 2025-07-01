#include "std_include.h"
#include "curand_kernel.h"
#include "indexer.h"
#include "selfPropelledParticleWithEdgeFriction.cuh"
#include <stdio.h>  // added 06-29-2025

__global__ void initRowPtr_kernel(int* row_ptr, int Nvertices)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2*Nvertices+1) return;

    row_ptr[idx] = 8*idx;
    }

__global__ void init_old_vn_kernel(int* old_vn, int Nvertices)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 3*Nvertices);

    old_vn[idx] = -1;
    }

__global__ void checkNeighborChange_kernel(
    int* old_vn,
    const int* __restrict__ new_vn,
    int Nvertices,
    int* changed)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nvertices) return;

    for (int k = 0; k < 3; ++k)
        {
        if (old_vn[3*idx+k] != new_vn[3*idx+k])
            {
            atomicExch(changed, 1);
            old_vn[3*idx+k] = new_vn[3*idx+k];
            }
        }
    }

__global__ void calculateForces_kernel(
    const double2* __restrict__ forces,
    const double2* __restrict__ motility,
    const double* __restrict__ cellDirectors,
    const int* __restrict__ vertexCellNeighbors,
    double* totalf_flat,
    int Nvertices)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nvertices) return;
    // Calculate the total force vector
    totalf_flat[2*idx] = forces[idx].x;
    totalf_flat[2*idx+1] = forces[idx].y;
    for (int k = 0; k < 3; ++k)
        {
        int cellidx = vertexCellNeighbors[3*idx+k];
        double v0 = motility[cellidx].x;
        double cosValue, sinValue;
        sincos(cellDirectors[cellidx], &sinValue, &cosValue);
        totalf_flat[2*idx] += v0*cosValue/3.0;
        totalf_flat[2*idx+1] += v0*sinValue/3.0;
        }
    }

__global__ void buildFrictionMatrixCSR_kernel(
    const double2* __restrict__ vertexPositions,
    const int* __restrict__ vertexNeighbors,
    periodicBoundaries Box,
    int Nvertices,
    double gamma_sub,
    double gamma_rel,
    int* col_idx,
    double* values)
    {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Nvertices) return;

    double Pthreshold = THRESHOLD;
    int offset = 0;
    double2 vec;
    double normsq, temp;
    double aux[3] = {0., 0., 0.};
    for (int k = 0; k < 3; ++k)
        {
        int j = vertexNeighbors[3*i+k];
        Box.minDist(vertexPositions[i], vertexPositions[j], vec);
        normsq = vec.x*vec.x + vec.y*vec.y;
        if (normsq < Pthreshold) normsq = Pthreshold;
        temp = vec.x*vec.x/normsq;
        aux[0] += temp;
        col_idx[8*(2*i) + 2*offset] = 2*j;
        values[8*(2*i) + 2*offset] = -gamma_rel*temp;
        temp = vec.x*vec.y/normsq;
        aux[1] += temp;
        col_idx[8*(2*i) + 2*offset + 1] = 2*j + 1;
        values[8*(2*i) + 2*offset + 1] = -gamma_rel*temp;
        col_idx[8*(2*i+1) + 2*offset] = 2*j;
        values[8*(2*i+1) + 2*offset] = -gamma_rel*temp;
        temp = vec.y*vec.y/normsq;
        aux[2] += temp;
        col_idx[8*(2*i+1) + 2*offset + 1] = 2*j + 1;
        values[8*(2*i+1) + 2*offset + 1] = -gamma_rel*temp;
        ++offset;
        }
    col_idx[8*(2*i) + 2*offset] = 2*i;
    values[8*(2*i) + 2*offset] = gamma_sub + gamma_rel*aux[0];
    col_idx[8*(2*i) + 2*offset + 1] = 2*i + 1;
    values[8*(2*i) + 2*offset + 1] = gamma_rel*aux[1];
    col_idx[8*(2*i+1) + 2*offset] = 2*i;
    values[8*(2*i+1) + 2*offset] = gamma_rel*aux[1];
    col_idx[8*(2*i+1) + 2*offset + 1] = 2*i + 1;
    values[8*(2*i+1) + 2*offset + 1] = gamma_sub + gamma_rel*aux[2];
    }

__global__ void calculate_vertex_displacements_from_velocities_kernel(
                                        double2* displacements,
                                        double2* vertexVelocities,
                                        const double* __restrict__ velocity_flat,
                                        int Nvertices,
                                        double deltaT,
                                        int Timestep)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Nvertices)
        return;

    vertexVelocities[idx].x = velocity_flat[2*idx];
    vertexVelocities[idx].y = velocity_flat[2*idx+1];
    // update displacements
    displacements[idx].x = deltaT*vertexVelocities[idx].x;
    displacements[idx].y = deltaT*vertexVelocities[idx].y;
    }

// copied from selfPropelledCellVertexDynamics.cu
__global__ void rotate_directors_kernel(
                                        double  *d_cellDirectors,
                                        curandState *d_curandRNGs,
                                        const double2 *motility,  // made const 06-29-2025
                                        double  deltaT,
                                        int      Ncells)
    {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= Ncells)
        return;

    //get the per-cell RNG, rotate the director, return the RNG
    curandState_t randState;
    randState=d_curandRNGs[idx];
    d_cellDirectors[idx] += cur_norm(&randState)*sqrt(2.0*deltaT*motility[idx].y);
    d_curandRNGs[idx] = randState;
    };

bool gpu_initRowPtr(int* row_ptr, int Nvertices)
    {
    int blockSize = 128;
    int nBlocks = (2*Nvertices) / blockSize + 1;
    initRowPtr_kernel<<<nBlocks,blockSize>>>(row_ptr, Nvertices);
    return true;

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
    }

bool gpu_init_old_vn(int* old_vn, int Nvertices)
    {
    int blockSize = 128;
    int nBlocks = (3*Nvertices + blockSize - 1) / blockSize;
    init_old_vn_kernel<<<nBlocks,blockSize>>>(old_vn, Nvertices);
    return true;
    }

bool gpu_spp_cellVertex_friction_eom_integration(
                            int* new_vn,
                            int* old_vn,
                            int nnz,
                            const double2* vertexPositions,
                            const double2* forces,
                            double2* vertexVelocities,
                            double* velocity_flat,
                            double* totalf_flat,
                            double2* displacements,
                            const double2 *motility,
                            double* cellDirectors,
                            int* vertexCellNeighbors,
                            int* d_col_idx,
                            double* d_values,
                            curandState* RNGs,
                            int Nvertices,
                            int Ncells,
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
    int nBlocks = (Nvertices + blockSize - 1) / blockSize;
    // check neighbor change
    cudaMemset(d_neigh_change, 0, sizeof(int));
    checkNeighborChange_kernel<<<nBlocks,blockSize>>>(old_vn,new_vn,Nvertices,d_neigh_change);
    int h_neigh_change = 0;
    cudaMemcpy(&h_neigh_change, d_neigh_change, sizeof(int), cudaMemcpyDeviceToHost);
    buildFrictionMatrixCSR_kernel<<<nBlocks,blockSize>>>(
        vertexPositions,
        new_vn,
        Box,
        Nvertices,
        gamma_sub,
        gamma_rel,
        d_col_idx,
        d_values);
    if (h_neigh_change != 0)
        {
        // Symbolic analysis (pattern only)
        cudssExecute(handle,
            CUDSS_PHASE_ANALYSIS,
            config,
            data,
            A, x, b);
        }

    // calculate the total forces vector, which automatically update b
    calculateForces_kernel<<<nBlocks,blockSize>>>(forces, motility, cellDirectors, vertexCellNeighbors, totalf_flat, Nvertices);
    // Numeric factorization
    cudssExecute(handle,
            CUDSS_PHASE_FACTORIZATION,
            config,
            data,
            A, x, b);
    // Solve
    cudssExecute(handle,
            CUDSS_PHASE_SOLVE,
            config,
            data,
            A, x, b);
    // integration and update
    calculate_vertex_displacements_from_velocities_kernel<<<nBlocks,blockSize>>>(
        displacements,
        vertexVelocities,
        velocity_flat,
        Nvertices,
        deltaT,
        Timestep);
    if (Ncells < 128) blockSize = 32;
    nBlocks = (Ncells + blockSize - 1) / blockSize;
    rotate_directors_kernel<<<nBlocks,blockSize>>>(
        cellDirectors,
        RNGs,
        motility,
        deltaT,
        Ncells);

    return true;
    }
