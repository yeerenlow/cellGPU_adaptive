#include "selfPropelledParticleWithEdgeFriction.h"
#include "selfPropelledParticleWithEdgeFriction.cuh"
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

/*! \file selfPropelledParticleWithEdgeFriction.cpp */
selfPropelledParticleWithEdgeFriction::selfPropelledParticleWithEdgeFriction(int _N, double _gamma_rel, bool _useGPU)
        : selfPropelledParticleDynamics{_N, _useGPU}
        , gamma_rel{_gamma_rel}, nnz{28*_N}
    {
    if (_useGPU)
        {
        int maxRows = 2 * _N;
        // Allocate memory for the sparse matrix representation on the device
        cudaMalloc(&d_row_ptr, (maxRows + 1) * sizeof(int));
        cudaMemset(d_row_ptr, 0, sizeof(int));
        cudaMalloc(&d_col_idx, nnz * sizeof(int));
        cudaMalloc(&d_values, nnz * sizeof(double));
        cudaMalloc(&velocity_flat, maxRows * sizeof(double));
        cudaMalloc(&totalf_flat, maxRows * sizeof(double));
        cudaMalloc(&d_row_sizes, maxRows * sizeof(int));
        cudaMalloc(&d_neigh_change, sizeof(int));
        cudaMalloc(&old_nn, _N * sizeof(int));
        cudaMalloc(&old_n, 16 * _N * sizeof(int));
        gpu_init_old_neighbors(old_nn, old_n, _N);
        // create and initialize cudss structures
        cudssCreate(&handle);
        cudssConfigCreate(&config);
        cudssDataCreate(handle, &data);
        cudssMatrixCreateCsr(
            &A,
            /*rows=*/2*_N,
            /*cols=*/2*_N,
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
            /*rows=*/2*Ndof,
            /*cols=*/1,
            /*leading dimension*/2*Ndof,
            totalf_flat,
            CUDA_R_64F,  // value type
            CUDSS_LAYOUT_COL_MAJOR);
        cudssMatrixCreateDn(
            &x,
            /*rows=*/2*Ndof,
            /*cols=*/1,
            /*leading dimension*/2*Ndof,
            velocity_flat,
            CUDA_R_64F,  // value type
            CUDSS_LAYOUT_COL_MAJOR);
        //Index2D n_idx = activeModel->n_idx;
        }
    }

void selfPropelledParticleWithEdgeFriction::computeFrictionMatrix(Eigen::SparseMatrix<double>& mat)
    {
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(28*Ndof);

    double Pthreshold = THRESHOLD;  // added 02-06-2025

    ArrayHandle<int> h_nn(activeModel->neighborNum,access_location::host,access_mode::read);
    ArrayHandle<int> h_n(activeModel->neighbors,access_location::host,access_mode::read);
    ArrayHandle<double2> h_p(activeModel->cellPositions,access_location::host,access_mode::read);
    for (int i = 0; i < Ndof; ++i)
        {
        double2 vec;
        double normsq;
        double aux[3] = {0., 0., 0.};
        double temp;
        for (int j = 0; j < h_nn.data[i]; ++j)
            {
            int k = h_n.data[activeModel->n_idx(j,i)];
            (activeModel->Box)->minDist(h_p.data[i], h_p.data[k], vec);
            normsq = vec.x*vec.x + vec.y*vec.y;
            if (normsq < Pthreshold) normsq = Pthreshold;  // added 02-06-2025
            temp = vec.y*vec.y/normsq;
            aux[0] += temp;
            tripletList.push_back(T(2*i, 2*k, -gamma_rel*temp));
            temp = -vec.x*vec.y/normsq;
            aux[1] += temp;
            tripletList.push_back(T(2*i, 2*k+1, -gamma_rel*temp));
            tripletList.push_back(T(2*i+1, 2*k, -gamma_rel*temp));
            temp = vec.x*vec.x/normsq;
            aux[2] += temp;
            tripletList.push_back(T(2*i+1, 2*k+1, -gamma_rel*temp));
            }
        tripletList.push_back(T(2*i, 2*i, gamma_sub + gamma_rel*aux[0]));
        tripletList.push_back(T(2*i, 2*i+1, gamma_rel*aux[1]));
        tripletList.push_back(T(2*i+1, 2*i, gamma_rel*aux[1]));
        tripletList.push_back(T(2*i+1, 2*i+1, gamma_sub + gamma_rel*aux[2]));
        }

    assert(tripletList.size() == 28*Ndof);  // added 05-19-2025

    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    }

/*!
The straightforward CPU implementation
*/
void selfPropelledParticleWithEdgeFriction::integrateEquationsOfMotionCPU()
    {
    Eigen::SparseMatrix<double> mat(2*Ndof,2*Ndof);
    computeFrictionMatrix(mat);
    Eigen::VectorXd f(2*Ndof), v(2*Ndof);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(mat);
    if (solver.info() != Eigen::Success)
        {
        printf("Eigen error at %s:%d\n", __FILE__, __LINE__);
        throw std::runtime_error("Matrix decomposition failed in selfPropelledParticleWithEdgeFriction");  // modified 06-28-2025
        }

    activeModel->computeForces();
    {// scope for array handles
    ArrayHandle<double2> h_f(activeModel->returnForces(),access_location::host,access_mode::read);
    ArrayHandle<double> h_cd(activeModel->cellDirectors);
    ArrayHandle<double2> h_v(activeModel->cellVelocities);
    ArrayHandle<double2> h_disp(displacements,access_location::host,access_mode::overwrite);
    ArrayHandle<double2> h_motility(activeModel->Motility,access_location::host,access_mode::read);

    for (int ii = 0; ii < Ndof; ++ii)
        {
        //displace according to current velocities and forces
        double v0i = h_motility.data[ii].x;
        //double Dri = h_motility.data[ii].y;
        // commented out 06-28-2025
        /*
        h_v.data[ii].x =  v0i * cos(h_cd.data[ii]);
        h_v.data[ii].y =  v0i * sin(h_cd.data[ii]);
        double2 Vcur = h_v.data[ii];
        */
        f(2*ii) = v0i * cos(h_cd.data[ii]) + h_f.data[ii].x;  // modified 06-28-2025
        f(2*ii+1) = v0i * sin(h_cd.data[ii]) + h_f.data[ii].y;  // modified 06-28-2025
        //h_disp.data[ii].x = deltaT*(Vcur.x + mu * h_f.data[ii].x);
        //h_disp.data[ii].y = deltaT*(Vcur.y + mu * h_f.data[ii].y);
        }

    v = solver.solve(f);
    if (solver.info() != Eigen::Success)
        {
        printf("Eigen error at %s:%d\n", __FILE__, __LINE__);
        throw std::runtime_error("Matrix solve failed in selfPropelledParticleWithEdgeFriction");  // modified 06-28-2025
        }

    for (int ii = 0; ii < Ndof; ++ii)
        {
        h_v.data[ii].x = v(2*ii);  // added 06-28-2025
        h_v.data[ii].y = v(2*ii+1);  // added 06-28-2025
        h_disp.data[ii].x = deltaT*v(2*ii);
        h_disp.data[ii].y = deltaT*v(2*ii+1);

        double Dri = h_motility.data[ii].y;
        //double2 Vcur = h_v.data[ii];  // commented out 06-28-2025
        double theta = h_cd.data[ii];
        //rotate the velocity vector a bit
        // commented out 06-28-2025
        /*
        if (!(Vcur.x == 0. && Vcur.y == 0.))
            {
            theta = atan2(Vcur.y,Vcur.x);
            };
        */
        double randomNumber = noise.getRealNormal();
        h_cd.data[ii] =theta+randomNumber*sqrt(2.0*deltaT*Dri);
        };
    }// end array handle scoping

    // commented out 06-28-2025
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
The GPU implementation of the self-propelled particle dynamics with edge friction
*/
void selfPropelledParticleWithEdgeFriction::integrateEquationsOfMotionGPU()
    {
    activeModel->computeForces();
        {//scope for array handles
        ArrayHandle<double2> d_p(activeModel->cellPositions,access_location::device,access_mode::read);
        ArrayHandle<double2> d_f(activeModel->returnForces(),access_location::device,access_mode::read);
        ArrayHandle<double> d_cd(activeModel->cellDirectors,access_location::device,access_mode::readwrite);
        ArrayHandle<double2> d_v(activeModel->cellVelocities,access_location::device,access_mode::readwrite);
        ArrayHandle<double2> d_disp(displacements,access_location::device,access_mode::overwrite);
        ArrayHandle<double2> d_motility(activeModel->Motility,access_location::device,access_mode::read);
        ArrayHandle<curandState> d_RNG(noise.RNGs,access_location::device,access_mode::readwrite);
        ArrayHandle<int> d_nn(activeModel->neighborNum,access_location::device,access_mode::read);
        ArrayHandle<int> d_n(activeModel->neighbors,access_location::device,access_mode::read);
        Index2D n_idx = activeModel->n_idx;
        periodicBoundaries Box = *(activeModel->Box);
        gpu_spp_friction_eom_integration(
                    d_nn.data,
                    d_n.data,
                    old_nn,
                    old_n,
                    nnz,
                    d_p.data,
                    d_f.data,
                    d_v.data,
                    velocity_flat,
                    totalf_flat,
                    d_disp.data,
                    d_motility.data,
                    d_cd.data,
                    d_row_ptr,
                    d_col_idx,
                    d_values,
                    d_row_sizes,
                    d_RNG.data,
                    Ndof,
                    n_idx,
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
        };//end array handle scope

    activeModel->moveDegreesOfFreedom(displacements);
    activeModel->enforceTopology();
    }
