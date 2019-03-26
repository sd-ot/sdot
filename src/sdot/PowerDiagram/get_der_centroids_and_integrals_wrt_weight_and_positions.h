#pragma once

#include "../Integration/SpaceFunctions/Constant.h"
#include "../Integration/FunctionEnum.h"
#include "../Support/ThreadPool.h"
#include "../Support/Assert.h"
#include <algorithm>
#include <vector>

namespace sdot {

/**
   We assume that grid has already been initialized by diracs.


*/
template<class TI,class TF,class Grid,class Bounds,class Pt,class Func>
int get_der_centroids_and_integrals_wrt_weight_and_positions( std::vector<TI> &m_offsets, std::vector<TI> &m_columns, std::vector<TF> &m_values, std::vector<TF> &v_values, Grid &grid, Bounds &bounds, const Pt *positions, const TF *weights, std::size_t nb_diracs, Func radial_func, bool stop_if_void = true ) {
    constexpr std::size_t dim = Grid::dim;
    constexpr std::size_t nupd = 1 + dim; // nb unknowns per dirac
    using TM = std::array<TF,nupd*nupd>;

    struct DataPerThread {
        DataPerThread( std::size_t approx_nb_diracs ) {
            row_items.reserve( 64 ); ///< room for tmp data (cleared for each new dirac)
            offsets  .reserve( approx_nb_diracs + 1 );
            columns  .reserve( 5 * approx_nb_diracs );
            values   .reserve( 5 * approx_nb_diracs );
        }

        std::vector<std::pair<TI,TM>> row_items; ///< [ ( column, values ) ]
        std::vector<TI>               offsets;
        std::vector<TI>               columns;
        std::vector<TM>               values;
    };

    int nb_threads = thread_pool.nb_threads();
    std::vector<DataPerThread> data_per_threads( nb_threads, nb_diracs / nb_threads );
    std::vector<std::pair<int,TI>> pos_in_loc_matrices( nb_diracs ); // num dirac => num_thread and num sub row

    v_values.clear();
    v_values.resize( nupd * nb_diracs, 0 );

    int err = grid.for_each_laguerre_cell( [&]( auto &lc, std::size_t num_dirac_0, int num_thread ) {
        DataPerThread &dpt = data_per_threads[ num_thread ];
        pos_in_loc_matrices[ num_dirac_0 ] = { num_thread, dpt.offsets.size() };
        dpt.row_items.resize( 0 );

        // get local row_items (sorted)
        TM der_0;
        for( auto &v : der_0 )
            v = TF( 0 );

        Pt d0_center = positions[ num_dirac_0 ];
        TF d0_weight = weights[ num_dirac_0 ];

        TF mass = 0;
        Pt centroid = TF( 0 );

        bounds.for_each_intersection( lc, [&]( auto &cp, SpaceFunctions::Constant<TF> space_func ) {
            cp.add_centroid_contrib( centroid, mass, radial_func.func_for_final_cp_integration(), space_func, d0_weight );

            TF coeff = 0.5 * space_func.coeff;
            cp.for_each_boundary_item( radial_func.func_for_final_cp_integration(), [&]( auto boundary_item ) {
                auto boundary_measure = boundary_item.measure;
                auto num_dirac_1 = boundary_item.id;

                if ( num_dirac_1 == TI( -1 ) )
                    return;
                if ( num_dirac_0 == num_dirac_1 ) {
                    der_0.back() += coeff * boundary_measure / sqrt( d0_weight );
                } else {
                    TI m_num_dirac_1 = num_dirac_1 % nb_diracs;
                    Pt d1_center = positions[ m_num_dirac_1 ];
                    if ( std::size_t nu = num_dirac_1 / nb_diracs )
                        TODO; // d1_center = transformation( _tranformations[ nu - 1 ], d1_center );

                    TF dist = norm_2( d0_center - d1_center );
                    TF b_der = coeff * boundary_measure / dist;

                    // area / weight
                    TM der_1;
                    for( auto &v : der_1 )
                        v = TF( 0 );
                    der_0[ nupd * dim + dim ] += b_der;
                    der_1[ nupd * dim + dim ] = - b_der;

                    // area / positions
                    for( std::size_t d = 0; d < dim; ++d ) {
                        TF a = boundary_item.points[ 0 ][ d ] - d0_center[ d ];
                        TF b = boundary_item.points[ 1 ][ d ] - d0_center[ d ];
                        der_0[ nupd * dim + d ] += coeff * boundary_measure * ( a + b ) / dist;
                    }
                    // area / positions
                    for( std::size_t d = 0; d < dim; ++d ) {
                        TF a = d1_center[ d ] - boundary_item.points[ 0 ][ d ];
                        TF b = d1_center[ d ] - boundary_item.points[ 1 ][ d ];
                        der_1[ nupd * dim + d ] += coeff * boundary_measure * ( a + b ) / dist;
                    }

                    dpt.row_items.emplace_back( m_num_dirac_1, der_1 );
                }
            }, weights[ num_dirac_0 ] );

            der_0.back() += cp.integration_der_wrt_weight( radial_func.func_for_final_cp_integration(), d0_weight );
        } );
        dpt.row_items.emplace_back( num_dirac_0, der_0 );
        std::sort( dpt.row_items.begin(), dpt.row_items.end() );

        for( std::size_t d = 0; d < dim; ++d )
            v_values[ nupd * num_dirac_0 + d ] = mass ? centroid[ d ] / mass : TF( 0 );
        v_values[ nupd * num_dirac_0 + dim ] = mass;


        // save them in local sub matrix
        dpt.offsets.push_back( dpt.columns.size() );
        for( std::size_t i = 0; i < dpt.row_items.size(); ++i ) {
            if ( i + 1 < dpt.row_items.size() && dpt.row_items[ i ].first == dpt.row_items[ i + 1 ].first ) {
                for( std::size_t d = 0; d < nupd * nupd; ++d )
                    dpt.row_items[ i + 1 ].second[ d ] += dpt.row_items[ i ].second[ d ];
                continue;
            }
            dpt.columns.push_back( dpt.row_items[ i ].first  );
            dpt.values .push_back( dpt.row_items[ i ].second );
        }
    }, bounds.englobing_convex_polyhedron(), positions, weights, nb_diracs, stop_if_void, radial_func.need_ball_cut() );
    if ( err )
        return err;

    // completion of local matrices
    std::size_t nnz = 0;
    for( DataPerThread &dpt : data_per_threads ) {
        dpt.offsets.push_back( dpt.columns.size() );
        nnz += dpt.columns.size();
    }

    // assembly
    m_offsets.resize( nupd * nb_diracs + 1 );
    m_columns.reserve( nnz );
    m_values .reserve( nnz );
    m_columns.resize( 0 );
    m_values .resize( 0 );
    for( std::size_t n = 0; n < nb_diracs; ++n ) {
        std::size_t lr = pos_in_loc_matrices[ n ].second;
        DataPerThread &dpt = data_per_threads[ pos_in_loc_matrices[ n ].first ];
        for( std::size_t d = 0; d < nupd; ++d ) {
            m_offsets[ nupd * n + d ] = m_columns.size();
            for( std::size_t k = dpt.offsets[ lr + 0 ]; k < dpt.offsets[ lr + 1 ]; ++k ) {
                for( std::size_t e = 0; e < nupd; ++e ) {
                    m_columns.push_back( nupd * dpt.columns[ k ] + e );
                    m_values.push_back( dpt.values[ k ][ nupd * d + e ] );
                }
            }
        }
    }
    m_offsets.back() = m_columns.size();

    return 0;
}

template<class TI,class TF,class Grid,class Bounds,class Pt>
int get_der_centroids_and_integrals_wrt_weight_and_positions( std::vector<TI> &m_offsets, std::vector<TI> &m_columns, std::vector<TF> &m_values, std::vector<TF> &v_values, Grid &grid, Bounds &bounds, const Pt *positions, const TF *weights, std::size_t nb_diracs ) {
    return get_der_centroids_and_integrals_wrt_weight_and_positions( m_offsets, m_columns, m_values, v_values, grid, bounds, positions, weights, nb_diracs, FunctionEnum::Unit() );
}

} // namespace sdot
