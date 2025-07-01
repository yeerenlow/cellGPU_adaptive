#include "selfPropelledCellVertexWithEdgeFriction.h"
#include "selfPropelledCellVertexWithEdgeFriction.cuh"
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

/*! \file selfPropelledCellVertexWithEdgeFriction.cpp */
selfPropelledCellVertexWithEdgeFriction::selfPropelledCellVertexWithEdgeFriction(int _Ncells, int _Nvertices, double _gamma_rel, bool _useGPU)
        : selfPropelledCellVertexDynamics{_Ncells, _Nvertices}
        , gamma_rel{_gamma_rel}, nnz{16*_Nvertices}
    {
    if (_useGPU)
        {
        int maxRows = 2 * _Nvertices;
        // Allocate memory for the sparse matrix representation on the device
        cudaMalloc(&d_row_ptr, (maxRows + 1) * sizeof(int));
        gpu_initRowPtr(d_row_ptr, Nvertices);
        //cudaMemset(d_row_ptr, 0, sizeof(int));
        cudaMalloc(&d_col_idx, nnz * sizeof(int));
        cudaMalloc(&d_values, nnz * sizeof(double));
        cudaMalloc(&velocity_flat, maxRows * sizeof(double));
        cudaMalloc(&totalf_flat, maxRows * sizeof(double));
        //cudaMalloc(&d_row_sizes, maxRows * sizeof(int));
        cudaMalloc(&d_neigh_change, sizeof(int));
        //cudaMalloc(&old_nn, _N * sizeof(int));
        cudaMalloc(&old_vn, 3 * _Nvertices * sizeof(int));
        gpu_init_old_vn(old_vn, _Nvertices);
        // create and initialize cudss structures
        cudssCreate(&handle);
        cudssConfigCreate(&config);
        cudssDataCreate(handle, &data);
        cudssMatrixCreateCsr(
            &A,
            /*rows=*/2*_Nvertices,
            /*cols=*/2*_Nvertices,
            nnz,
            d_row_ptr,
            NULL,
            d_col_idx,
            d_values,
            CUDA_R_32I,  // index type
            CUDA_R_64F,  // value type
            CUDSS_MTYPE_SPD,  // Matrix type symmetric positive definite
            CUDSS_MVIEW_FULL,  // Matrix view type
            CUDSS_BASE_ZERO
        );
        cudssMatrixCreateDn(
            &b,
            /*rows=*/2*Nvertices,
            /*cols=*/1,
            /*leading dimension*/2*Nvertices,
            totalf_flat,
            CUDA_R_64F,  // value type
            CUDSS_LAYOUT_COL_MAJOR);
        cudssMatrixCreateDn(
            &x,
            /*rows=*/2*Nvertices,
            /*cols=*/1,
            /*leading dimension*/2*Nvertices,
            velocity_flat,
            CUDA_R_64F,  // value type
            CUDSS_LAYOUT_COL_MAJOR);
        //Index2D n_idx = activeModel->n_idx;
        }
    }

void selfPropelledCellVertexWithEdgeFriction::computeFrictionMatrix(Eigen::SparseMatrix<double>& mat)
    {
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(16*Nvertices);

    double Pthreshold = THRESHOLD;  // added 02-06-2025

    ArrayHandle<int> h_vn(activeModel->vertexNeighbors,access_location::host,access_mode::read);
    ArrayHandle<double2> h_v(activeModel->vertexPositions,access_location::host,access_mode::read);
    for (int i = 0; i < Nvertices; ++i)
        {
        double2 vec;
        double normsq;
        double aux[3] = {0., 0., 0.};
        double temp;
        for (int j = 0; j < 3; ++j)
            {
            (activeModel->Box)->minDist(h_v.data[i],h_v.data[h_vn.data[3*i+j]],vec);
            normsq = vec.x*vec.x + vec.y*vec.y;
            if (normsq < Pthreshold) normsq = Pthreshold;  // added 02-06-2025
            temp = vec.x*vec.x/normsq;
            aux[0] += temp;
            tripletList.push_back(T(2*i, 2*h_vn.data[3*i+j], -gamma_rel*temp));
            temp = vec.x*vec.y/normsq;
            aux[1] += temp;
            tripletList.push_back(T(2*i, 2*h_vn.data[3*i+j]+1, -gamma_rel*temp));
            tripletList.push_back(T(2*i+1, 2*h_vn.data[3*i+j], -gamma_rel*temp));
            temp = vec.y*vec.y/normsq;
            aux[2] += temp;
            tripletList.push_back(T(2*i+1, 2*h_vn.data[3*i+j]+1, -gamma_rel*temp));
            }
        tripletList.push_back(T(2*i, 2*i, gamma_sub + gamma_rel*aux[0]));
        tripletList.push_back(T(2*i, 2*i+1, gamma_rel*aux[1]));
        tripletList.push_back(T(2*i+1, 2*i, gamma_rel*aux[1]));
        tripletList.push_back(T(2*i+1, 2*i+1, gamma_sub + gamma_rel*aux[2]));
        }

    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    }

/*!
The straightforward CPU implementation
*/
void selfPropelledCellVertexWithEdgeFriction::integrateEquationsOfMotionCPU()
    {
    /*
    Eigen::SparseMatrix<double> mat(Nvertices,Nvertices);
    computeFrictionMatrixEigen(mat);
    Eigen::VectorXd f_x(Nvertices), f_y(Nvertices), v_x(Nvertices), v_y(Nvertices);
    */  // modified 01-19-2025
    Eigen::SparseMatrix<double> mat(2*Nvertices,2*Nvertices);
    computeFrictionMatrix(mat);
    Eigen::VectorXd f(2*Nvertices), v(2*Nvertices);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(mat);
    if (solver.info() != Eigen::Success)
        {
        printf("Eigen error at %s:%d\n", __FILE__, __LINE__);
        throw std::runtime_error("Matrix decomposition failed in selfPropelledCellVertexWithEdgeFriction");  // modified 06-29-2025
        }

    activeModel->computeForces();
    { // scope for arrayHandles
    ArrayHandle<double2> h_f(activeModel->returnForces(),access_location::host,access_mode::read);
    ArrayHandle<double> h_cd(activeModel->cellDirectors,access_location::host,access_mode::readwrite);
    ArrayHandle<double2> h_v(activeModel->vertexVelocities,access_location::host,access_mode::readwrite);  // added 06-29-2025
    ArrayHandle<double2> h_disp(displacements,access_location::host,access_mode::overwrite);
    ArrayHandle<double2> h_motility(activeModel->Motility,access_location::host,access_mode::read);
    ArrayHandle<int> h_vcn(activeModel->vertexCellNeighbors,access_location::host,access_mode::read);

    double directorx,directory;
    for (int i = 0; i < Nvertices; ++i)
        {
        double v1 = h_motility.data[h_vcn.data[3*i]].x;
        double v2 = h_motility.data[h_vcn.data[3*i+1]].x;
        double v3 = h_motility.data[h_vcn.data[3*i+2]].x;
        //for uniform v0, the vertex director is the straight average of the directors of the cell neighbors
        directorx  = v1*cos(h_cd.data[ h_vcn.data[3*i] ]);
        directorx += v2*cos(h_cd.data[ h_vcn.data[3*i+1] ]);
        directorx += v3*cos(h_cd.data[ h_vcn.data[3*i+2] ]);
        directorx /= 3.0;
        directory  = v1*sin(h_cd.data[ h_vcn.data[3*i] ]);
        directory += v2*sin(h_cd.data[ h_vcn.data[3*i+1] ]);
        directory += v3*sin(h_cd.data[ h_vcn.data[3*i+2] ]);
        directory /= 3.0;
        //compute -dE/dr + active forces
        //f_x(i) = directorx + h_f.data[i].x;
        //f_y(i) = directory + h_f.data[i].y;
        f(2*i) = directorx + h_f.data[i].x;
        f(2*i+1) = directory + h_f.data[i].y;  // modified 01-19-2025
        };

    /*
    v_x = solver.solve(f_x);
    if (solver.info() != Eigen::Success)
        {
        printf("Eigen error at %s:%d\n", __FILE__, __LINE__);
        throw std::runtime_error("Eigen error");
        }
    v_y = solver.solve(f_y);
    */  // modified 01-19-2025
    v = solver.solve(f);
    if (solver.info() != Eigen::Success)
        {
        printf("Eigen error at %s:%d\n", __FILE__, __LINE__);
        throw std::runtime_error("Matrix solve failed in selfPropelledCellVertexWithEdgeFriction");  // modified 06-29-2025
        }

    for (int i = 0; i < Nvertices; ++i)
        {
        // calculate displacements
        //h_disp.data[i].x = deltaT*v_x(i);
        //h_disp.data[i].y = deltaT*v_y(i);
        h_v.data[i].x = v(2*i);  // added 06-29-2025
        h_v.data[i].y = v(2*i+1);  // added 06-29-2025
        h_disp.data[i].x = deltaT*v(2*i);
        h_disp.data[i].y = deltaT*v(2*i+1);  // modified 01-19-2025
        }
    //update cell directors
    for (int i = 0; i < Ncells; ++i)
        {
        double randomNumber = noise.getRealNormal();
        double Dr = h_motility.data[i].y;
        h_cd.data[i] += randomNumber*sqrt(2.0*deltaT*Dr);
        };
    } // end array handle scoping

    // added 02-01-2025
    // commented out 06-29-2025
    /*
    if (shear_rate != 0.0)
        {
        activeModel->applyShear(shear_rate*deltaT);
        (activeModel->Box)->addOffset(shear_rate*deltaT);
        }
    */

    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
    }

/*!
The GPU implementation of the self-propelled cell vertex dynamics with edge friction
*/
void selfPropelledCellVertexWithEdgeFriction::integrateEquationsOfMotionGPU()
    {
    activeModel->computeForces();

        {//scope for array handles
        ArrayHandle<double2> d_p(activeModel->vertexPositions,access_location::device,access_mode::read);
        ArrayHandle<double2> d_f(activeModel->returnForces(),access_location::device,access_mode::read);
        ArrayHandle<double> d_cd(activeModel->cellDirectors,access_location::device,access_mode::readwrite);
        ArrayHandle<double2> d_v(activeModel->vertexVelocities,access_location::device,access_mode::readwrite);
        ArrayHandle<double2> d_disp(displacements,access_location::device,access_mode::overwrite);
        ArrayHandle<double2> d_motility(activeModel->Motility,access_location::device,access_mode::read);
        ArrayHandle<curandState> d_RNG(noise.RNGs,access_location::device,access_mode::readwrite);
        ArrayHandle<int> d_vn(activeModel->vertexNeighbors,access_location::device,access_mode::read);
        ArrayHandle<int> d_vcn(activeModel->vertexCellNeighbors,access_location::device,access_mode::read);
        //Index2D n_idx = activeModel->n_idx;
        periodicBoundaries Box = *(activeModel->Box);
        gpu_spp_cellVertex_friction_eom_integration(
                    d_vn.data,
                    old_vn,
                    nnz,
                    d_p.data,
                    d_f.data,
                    d_v.data,
                    velocity_flat,
                    totalf_flat,
                    d_disp.data,
                    d_motility.data,
                    d_cd.data,
                    d_vcn.data,
                    d_col_idx,
                    d_values,
                    d_RNG.data,
                    Nvertices,
                    Ncells,
                    Box,
                    deltaT,
                    Timestep,
                    gamma_sub,
                    gamma_rel,
                    handle,
                    config,
                    data,
                    A,
                    b,
                    x,
                    d_neigh_change);
        }

    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
    }
