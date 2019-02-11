#pragma once

// #include <boost/multiprecision/mpfr.hpp>
#include <vector>

/**
*/
class EigenSolver {
public:
    using TI = std::size_t;
    using TF = double; // boost::multiprecision::mpfr_float_100; //

    void  solve( std::vector<TF> &x, const std::vector<TI> &m_offsets, const std::vector<TI> &m_columns, const std::vector<TF> &m_values, const std::vector<TF> &v_values );

    int   nb_iters;
    TF    error;
};
