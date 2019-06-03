
#pragma once

#include "../Integration/SpaceFunctions/Constant.h"
#include "../Integration/FunctionEnum.h"
#include "../Support/CrossProdOfRanges.h"
#include "../Support/ThreadPool.h"
#include <vector>
#include <array>

namespace sdot {

/**
   We assume that grid has already been initialized by diracs
*/
template<class TF,class Grid,class Bounds,class Pt,class Func>
void get_image_integrals( TF *res, Grid &grid, Bounds &bounds, const Pt *positions, const TF *weights, std::size_t nb_diracs, const Func &radial_func, Pt min_pt, Pt max_pt, std::array<std::size_t,Grid::dim> nbp ) {
    constexpr std::size_t dim = Grid::dim;
    using TI = std::size_t;
    using std::min;
    using std::max;

    std::size_t n = nbp[ 0 ];
    for( std::size_t d = 1; d < nbp.size(); ++d )
        n *= nbp[ d ];
    std::vector<TF> tmp_res( n * ( dim + 1 ) * thread_pool.nb_threads(), 0 );

    grid.for_each_laguerre_cell( [&]( auto &lc, auto num_dirac, int num_thread ) {
        TF *ptr_res = tmp_res.data() + n * ( dim + 1 ) * num_thread;
        bounds.for_each_intersection( lc, [&]( auto &cp, SpaceFunctions::Constant<TF> space_func ) {
            using CP = typename Grid::CP;

            // find min_y, max_y
            Pt ps;
            std::array<TI,dim> min_i;
            std::array<TI,dim> max_i;
            Pt min_pos = cp.min_position();
            Pt max_pos = cp.max_position();
            for( std::size_t d = 0; d < dim; ++d ) {
                min_i[ d ] = ( min_pos[ d ] - min_pt[ d ] ) * nbp[ d ] / ( max_pt[ d ] - min_pt[ d ] );
                max_i[ d ] = ( max_pos[ d ] - min_pt[ d ] ) * nbp[ d ] / ( max_pt[ d ] - min_pt[ d ] );
                min_i[ d ] = max( TI( 0 ), min_i[ d ] );
                max_i[ d ] = min( nbp[ d ], max_i[ d ] + 1 );
                ps[ d ] = ( max_pt[ d ] - min_pt[ d ] ) / nbp[ d ];
            }


            // for each pixel
            CP ccp;
            CrossProdOfRanges<std::size_t,dim> cr( min_i, max_i );
            cr.for_each( [&]( auto p ) {
                Pt pf;
                TI off_pix = 0;
                for( std::size_t d = 0, acc = 1; d < dim; ++d ) {
                    off_pix += acc * p[ d ];
                    pf[ d ] = p[ d ];
                    acc *= nbp[ d ];
                }
                off_pix *= dim + 1;

                ccp = { typename CP::Box{ min_pt + ps * ( pf + TF( 0 ) ), min_pt + ps * ( pf + TF( 1 ) ) }, typename CP::CI( -1 ) };
                ccp.intersect_with( cp );

                TF m = space_func.coeff * ccp.measure( radial_func.func_for_final_cp_integration(), weights[ num_dirac ] );
                for( std::size_t d = 0; d < dim; ++d )
                    ptr_res[ off_pix + d ] += m * positions[ num_dirac ][ d ];
                ptr_res[ off_pix + dim ] += m;
            } );
        } );
    }, bounds.englobing_convex_polyhedron(), positions, weights, nb_diracs, false, radial_func.need_ball_cut() );

    // save in res
    for( std::size_t i = 0; i < n * ( dim + 1 ); ++i )
        res[ i ] = 0;
    for( int nt = 0; nt < thread_pool.nb_threads(); ++nt )
        for( std::size_t i = 0; i < n * ( dim + 1 ); ++i )
            res[ i ] += tmp_res[ i + n * ( dim + 1 ) * nt ];
}

} // namespace sdot
