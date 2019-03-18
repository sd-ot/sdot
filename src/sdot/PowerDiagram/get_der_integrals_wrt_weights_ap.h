#pragma once

#include "get_integrals.h"
#include <vector>

namespace sdot {

/**
  Approximate derivatives
*/
template<class TI,class TF,class Grid,class Bounds,class Diracs>
void get_der_measures_ap( std::vector<TI> &m_offsets, std::vector<TI> &m_columns, std::vector<TF> &m_values, std::vector<TF> &measures, Grid &grid, Bounds &bounds, const Diracs &diracs, TF epsilon = TF( 1e-10 ) ) {
    // rhs
    get_measures( measures, grid, bounds, diracs );

    // matrix
    m_offsets.resize( diracs.size() + 1 );
    m_columns.resize( 0 );
    m_values .resize( 0 );
    std::vector<TF> new_measures( diracs.size() );
    std::vector<std::vector<TF>> values( diracs.size() );
    for( std::size_t num_dirac_0 = 0; num_dirac_0 < diracs.size(); ++num_dirac_0 ) {
        Diracs new_diracs = diracs;
        new_diracs[ num_dirac_0 ].weight += epsilon;
        get_measures( new_measures, grid, bounds, new_diracs );

        m_offsets[ num_dirac_0 ] = m_columns.size();
        for( std::size_t num_dirac_1 = 0; num_dirac_1 < diracs.size(); ++num_dirac_1 ) {
            if( TF d = ( new_measures[ num_dirac_1 ] - measures[ num_dirac_1 ] ) / epsilon ) {
                m_columns.push_back( num_dirac_1 );
                m_values .push_back( d );
            }
        }
    }
    m_offsets.back() = m_columns.size();
}

} // namespace sdot
