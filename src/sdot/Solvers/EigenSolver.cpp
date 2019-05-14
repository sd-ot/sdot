#include <eigen3/Eigen/SparseCholesky>
#include <eigen3/Eigen/Eigenvalues>
#include "EigenSolver.h"

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -ffast-math
//// nsmake cpp_flag -O3

void EigenSolver::solve( std::vector<TF> &x, const std::vector<TI> &m_offsets, const std::vector<TI> &m_columns, const std::vector<TF> &m_values, const std::vector<TF> &v_values ) {
    using EV = Eigen::Matrix<TF,Eigen::Dynamic,1>;
    using EM = Eigen::SparseMatrix<TF,0,std::ptrdiff_t>;

    Eigen::Map<const EM> M(
         v_values.size(),
         v_values.size(),
         m_values.size(),
         (const std::ptrdiff_t *)m_offsets.data(),
         (const std::ptrdiff_t *)m_columns.data(),
         m_values.data()
    );

    //    Eigen::Matrix<TF,Eigen::Dynamic,Eigen::Dynamic> m( M );
    //    PN( m );
    //    Eigen::EigenSolver<Eigen::MatrixXd> es{ m, false };
    //    P( es.eigenvalues().real().maxCoeff(), es.eigenvalues().real().minCoeff() );

    // fact
    Eigen::SimplicialCholesky<EM> C;
    C.compute( M );

    // solve and update the weights
    EV V( v_values.size() );
    for( std::size_t i = 0; i < v_values.size(); ++i )
        V[ i ] = v_values[ i ];
    EV D = C.solve( V );

    //
    x.resize( D.size() );
    for( std::size_t i = 0; i < v_values.size(); ++i )
        x[ i ] = D[ i ];
}

