#ifndef SELF_PROPELLED_CELL_VERTEX_WITH_EDGE_FRICTION_CUH
#define SELF_PROPELLED_CELL_VERTEX_WITH_EDGE_FRICTION_CUH

#include "selfPropelledCellVertexWithEdgeFriction.h"

bool gpu_initRowPtr(int* row_ptr, int Nvertices);

bool gpu_init_old_vn(int* old_vn, int Nvertices);

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
                            int* d_neigh_change);

#endif

