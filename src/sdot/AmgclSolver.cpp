#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>

#include "system/Stream.h"
#include "AmgclSolver.h"

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -ffast-math
//// nsmake cpp_flag -O6
//// nsmake lib_flag -O6

//// nsmake cpp_flag -fopenmp
//// nsmake lib_name gomp

void AmgclSolver::solve( std::vector<TF> &x, const std::vector<TI> &m_offsets, const std::vector<TI> &m_columns, const std::vector<TF> &m_values, const std::vector<TF> &v_values ) {
    using Backend = amgcl::backend::builtin<double>;

    using Solve = amgcl::make_solver<
        // Use AMG as preconditioner:
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
            >,
        // And BiCGStab as iterative solver:
        amgcl::solver::bicgstab<Backend>
        // amgcl::solver::cg<Backend>
    >;

    TI n = v_values.size();
    Solve solve( std::tie( n, m_offsets, m_columns, m_values ) );

    std::tie( nb_iters, error ) = solve( v_values, x );
    // P( error, nb_iters );
}

