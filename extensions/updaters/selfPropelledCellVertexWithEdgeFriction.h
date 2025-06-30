#ifndef SELF_PROPELLED_CELL_VERTEX_WITH_EDGE_FRICTION_H
#define SELF_PROPELLED_CELL_VERTEX_WITH_EDGE_FRICTION_H

#include "selfPropelledCellVertexDynamics.h"
#include "std_include.h"
#include <cuda_runtime.h>
#include <cudss.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <math.h>
#include <Eigen/SparseCore>
#include <library_types.h>

class selfPropelledCellVertexWithEdgeFriction : public selfPropelledCellVertexDynamics
{
public:
    // Constructor
    selfPropelledCellVertexWithEdgeFriction(int _Ncells, int _Nvertices, double _gamma_rel = 1.0, bool _useGPU = true);
    ~selfPropelledCellVertexWithEdgeFriction()
    {
        if (d_row_ptr) cudaFree(d_row_ptr);
        if (d_col_idx) cudaFree(d_col_idx);
        if (d_values) cudaFree(d_values);
        if (velocity_flat) cudaFree(velocity_flat);
        if (totalf_flat) cudaFree(totalf_flat);
        //if (d_row_sizes) cudaFree(d_row_sizes);
        if (d_neigh_change) cudaFree(d_neigh_change);
        //if (old_nn) cudaFree(old_nn);
        if (old_vn) cudaFree(old_vn);
        cudssMatrixDestroy(A);
        cudssMatrixDestroy(x);
        cudssMatrixDestroy(b);
        cudssDataDestroy(handle, data);
        cudssConfigDestroy(config);
        cudssDestroy(handle);
    }
    // Add public methods and members here
    virtual void integrateEquationsOfMotionCPU();
    virtual void integrateEquationsOfMotionGPU();
    void setGammaSub(double _gamma_sub) {gamma_sub = _gamma_sub;};
    void setGammaRel(double _gamma_rel) {gamma_rel = _gamma_rel;};
protected:
    // Add protected members here
    virtual void computeFrictionMatrix(Eigen::SparseMatrix<double> &mat);
    int *d_neigh_change;  // have neighbors changed?
    //int *d_row_sizes = nullptr;  // row sizes for sparse matrix
    int *d_row_ptr = nullptr;  // row pointer for sparse matrix
    int *d_col_idx = nullptr;  // column indices for sparse matrix
    int nnz;  // number of non-zero entries in sparse matrix
    double *d_values = nullptr;  // values for sparse matrix
    double *velocity_flat = nullptr;  // Flattened velocity array for GPU
    double *totalf_flat = nullptr;  // Flattened total force array for GPU
    double gamma_rel;
    double gamma_sub = 1.;
    //int *old_nn;  // old number of neighbors
    int *old_vn;  // old vertex neighbors
    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t data;
    cudssMatrix_t A,b,x;
};

#endif
