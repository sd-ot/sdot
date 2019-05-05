#include "../PowerDiagram/get_der_integrals_wrt_weights.h"
#include "../PowerDiagram/get_centroids.h"
#include "OptimalTransportSolver.h"
// #include "traversal_cgal.h"
#include "../PowerDiagram/get_integrals.h"
#include "../Display/VtkOutput.h"
//#include "AmgclSolver.h"
#include "EigenSolver.h"
#include "../Support/Stream.h"
#include "../Support/Tick.h"
#include "../Support/Mpi.h"

namespace sdot {

template<class Grid, class Bounds>
OptimalTransportSolver<Grid, Bounds>::OptimalTransportSolver( Grid *grid, Bounds *bounds ) : bounds( *bounds ), grid( *grid ) {
    max_nb_iter = 150;
}

template<class Grid, class Bounds> template<class RadialFunc>
void OptimalTransportSolver<Grid, Bounds>::get_centroids( Pt *centroids, const Pt *positions, TF *weights, TI nb_diracs, RadialFunc rf ) {
    sdot::get_centroids( grid, bounds, positions, weights, nb_diracs, rf, [&]( auto c, auto m, auto n ) {
        centroids[ n ] = c;
    } );
}

template<class Grid, class Bounds> template<class RadialFunc>
void OptimalTransportSolver<Grid, Bounds>::solve( const Pt *positions, TF *weights, TI nb_diracs, TF *masses, RadialFunc rf ) {
    using std::max;
    using std::abs;

    old_weights.resize( nb_diracs );
    for( std::size_t i = 0; i < nb_diracs; ++i )
        old_weights[ i ] = weights[ i ];

    // max_nb_iter = 4;

    for( std::size_t num_iter = 0; num_iter < max_nb_iter; ++num_iter ) {
        // grid
        auto t0 = Tick::get_time();
        grid.update( positions, weights, nb_diracs, num_iter == 0, true );
        timings_grid.push_back( Tick::elapsed_since( t0 ) );

        // P( grid.check_sanity( positions ) );

        //        VtkOutput<1> vo_grid( { "num" } );
        //        grid.display( vo_grid );
        //        vo_grid.save( "vtk/grid.vtk" );
        //        VtkOutput<2> vtk_output( { "weight", "num" } );
        //        display( vtk_output, positions, weights, nb_diracs );
        //        vtk_output.save( "vtk/pd.vtk" );

        //        v_values.resize( nb_diracs );
        //        t0 = Tick::get_time();
        //        get_integrals( v_values.data(), grid, bounds, positions, weights, nb_diracs );
        //        auto t0_der = Tick::elapsed_since( t0 );
        // P( v_values );

        // der
        t0 = Tick::get_time();
        int error = get_der_integrals_wrt_weights( m_offsets, m_columns, m_values, v_values, grid, bounds, positions, weights, nb_diracs, rf );
        auto t0_der = Tick::elapsed_since( t0 );
        if ( m_values.size() && rf.need_ball_cut() == 0 )
            m_values[ 0 ] *= 2;

        //        if ( mpi->size() == 1 ) {
        //            t0 = Tick::get_time();
        //            traversal_cgal( reinterpret_cast<const TF *>( positions ), weights, nb_diracs );
        //            timings_cgal.push_back( Tick::elapsed_since( t0 ) );
        //        }

        // go back if pb
        if ( error ) {
            P( "error" );
            TF ratio = 0.1;
            for( std::size_t i = 0; i < nb_diracs; ++i )
                weights[ i ] = ( 1 - ratio ) * old_weights[ i ] + ratio * weights[ i ];
            timings_solve.push_back( 0 );
            continue;
        }
        for( std::size_t i = 0; i < nb_diracs; ++i )
            old_weights[ i ] = weights[ i ];

        timings_der.push_back( t0_der );

        //        TF vol = 0;
        //        for( std::size_t i = 0; i < nb_diracs; ++i )
        //            vol += v_values[ i ];
        //        PMPI_0( mpi->reduction( vol, [&]( double a, double b ) { return a + b; } ) );

        if ( max_nb_iter == 1 )
            break;

        // target
        for( std::size_t i = 0; i < nb_diracs; ++i )
            v_values[ i ] -= masses[ i ];

        // solve
        EigenSolver es;
        // AmgclSolver es;
        t0 = Tick::get_time();
        es.solve( dw, m_offsets, m_columns, m_values, v_values );
        timings_solve.push_back( Tick::elapsed_since( t0 ) );

        TF mdw = 0;
        for( std::size_t i = 0; i < nb_diracs; ++i ) {
            mdw = max( mdw, abs( dw[ i ] ) );
            weights[ i ] -= dw[ i ];
        }

        P( mdw );
        if ( mdw < 1e-6 )
            break;
    }

    //    P( timings_grid  );
    //    P( timings_der   );
    //    P( timings_cgal  );
    //    P( timings_solve );
}

template<class Grid, class Bounds> template<class VO,class RF>
void OptimalTransportSolver<Grid,Bounds>::display( VO &vtk_output, const Pt *positions, const TF *weights, TI nb_diracs, RF rf ) {
    grid.update( positions, weights, nb_diracs );
    grid.for_each_laguerre_cell( [&]( auto &lc, std::size_t num_dirac_0, int ) {
        //        domain.bounds.for_each_intersection( lc, [&]( auto &cp, auto space_func ) {
        //            cp.display( vtk_output, { ptr_weights[ num_dirac_0 ], TF( num_dirac_0 ) } );
        //        } );
        lc.display( vtk_output, { weights[ num_dirac_0 ], TF( num_dirac_0 ) } );
    }, bounds.englobing_convex_polyhedron(), positions, weights, nb_diracs, false, rf.need_ball_cut() );
}

template<class Grid, class Bounds> template<class VO>
void OptimalTransportSolver<Grid,Bounds>::display_orig_pts( VO &vtk_output, const Pt *positions, const TF *weights, TI nb_diracs ) {
    using std::min;
    using std::max;

    std::vector<TF> new_weights( nb_diracs, 1 );

    TF min_w = + std::numeric_limits<TF>::max();
    TF max_w = - std::numeric_limits<TF>::max();
    for( std::size_t i = 0; i < nb_diracs; ++i ) {
        min_w = min( min_w, weights[ i ] );
        max_w = max( max_w, weights[ i ] );
    }

    grid.update( positions, new_weights.data(), nb_diracs );
    grid.for_each_laguerre_cell( [&]( auto &lc, TI num_dirac_0 ) {
        vtk_output.mutex.lock();
        lc.for_each_simplex( [&]( TI num_dirac_1, TI num_dirac_2 ) {
            if ( int( num_dirac_1 ) < 0 || int( num_dirac_2 ) < 0 )
                return;
            ASSERT( num_dirac_0 < nb_diracs, "" );
            ASSERT( num_dirac_1 < nb_diracs, "" );
            ASSERT( num_dirac_2 < nb_diracs, "" );
            if ( num_dirac_0 < num_dirac_1 && num_dirac_0 < num_dirac_2 ) {
                vtk_output.add_polygon( {
                    Point3<TF>( positions[ num_dirac_0 ].x, positions[ num_dirac_0 ].y, weights[ num_dirac_0 ] - min_w ),
                    Point3<TF>( positions[ num_dirac_1 ].x, positions[ num_dirac_1 ].y, weights[ num_dirac_1 ] - min_w ),
                    Point3<TF>( positions[ num_dirac_2 ].x, positions[ num_dirac_2 ].y, weights[ num_dirac_2 ] - min_w )
                }, { ( weights[ num_dirac_0 ] + weights[ num_dirac_1 ] + weights[ num_dirac_2 ] ) / 3 } );
            }
        } );
        vtk_output.mutex.unlock();
    }, bounds.englobing_convex_polyhedron(), positions, new_weights.data(), nb_diracs );
}

//    template<class Diracs>
//    TF volume( const Diracs &diracs ) {
//        grid.init( diracs );

//        std::vector<TF> volumes( diracs.size() );
//        grid.for_each_laguerre_cell( [&]( CP &lc, std::size_t num ) {
//            bounds.for_each_intersection( lc, [&]( CP &cp, auto space_func ) {
//                volumes[ num ] += cp.integration( space_func );
//            } );
//        }, bounds.englobing_convex_polyhedron(), diracs );

//        TF res = 0;
//        for( auto v : volumes )
//            res += v;
//        return res;
//    }

} // namespace sdot
