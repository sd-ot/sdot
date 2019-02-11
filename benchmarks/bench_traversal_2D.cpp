#include "../src/PowerDiagram/Bounds/ConvexPolyhedronAssembly.h"
#include "../src/PowerDiagram/Visitors/ZGrid.h"
#include "../src/PowerDiagram/system/Tick.h"
#include "../src/PowerDiagram/VtkOutput.h"
#include <gtest/gtest.h>

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -ffast-math
//// nsmake cpp_flag -O5
//// nsmake lib_flag -O5

template<class Dirac>
void set_up_diracs( std::vector<Dirac> &diracs, std::string distribution, std::size_t nb_diracs ) {
    using TI = std::size_t;

    if ( distribution == "regular" ) {
        TI l = std::sqrt( nb_diracs );
        diracs.resize( l * l );
        for( TI i = 0, c = 0; i < l; ++i )
            for( TI j = 0; j < l; ++j )
                diracs[ c++ ] = { { ( j + 0.45 + 0.1 * rand() / RAND_MAX ) / l, ( i + 0.45 + 0.1 * rand() / RAND_MAX ) / l }, 1.0 };
        return;
    }

    if ( distribution == "regular_with_noise" ) {
        TI l = std::sqrt( nb_diracs );
        diracs.resize( l * l );
        double span = 0.5;
        for( TI i = 0, c = 0; i < l; ++i ) {
            for( TI j = 0; j < l; ++j ) {
                diracs[ c++ ] = { {
                    ( i + 0.5 + span * ( 1.0 * rand() / RAND_MAX - 0.5 ) ) / l,
                    ( j + 0.5 + span * ( 1.0 * rand() / RAND_MAX - 0.5 ) ) / l
                }, 1.0 };
            }
        }
        return;
    }

    if ( distribution == "random" ) {
        diracs.resize( nb_diracs );
        for( std::size_t i = 0; i < nb_diracs; ++i )
            diracs[ i ] = { { 0.1 + 0.8 * rand() / RAND_MAX, 0.1 + 0.8 * rand() / RAND_MAX }, 1.0 };
        return;
    }

    if ( distribution == "lines" ) {
        diracs.resize( 0 );
        for( std::size_t i = 0; i < nb_diracs / 2; ++i ) {
            double y = 1.0 * rand() / RAND_MAX;
            double x = 0.1 + 0.05 * rand() / RAND_MAX;
            double w = 1; // 0.5 + 0.05 * rand() / RAND_MAX;
            diracs.push_back( { { x, y }, w } );
        }
        for( std::size_t i = 0; i < nb_diracs / 2; ++i ) {
            double y = 1.0 * rand() / RAND_MAX;
            double x = 0.9 - 0.5 * y + 0.05 * rand() / RAND_MAX;
            double w = 1; // 0.5 + 0.05 * rand() / RAND_MAX;
            diracs.push_back( { { x, y }, w } );
        }

        return;
    }

    TODO;
}


template<class Pc>
void launch_bench( Pc, std::string distribution, int max_pts_per_cell, std::size_t nb_diracs, std::string vo_output = {} ) {
    using namespace sdot;

    using  TF    = typename Pc::TF;
    struct Dirac { Point2<TF> pos; TF weight; };
    using  ZGrid = Visitor::ZGrid<Pc>;
    using  LC    = typename ZGrid::CP;

    std::vector<Dirac> diracs;
    set_up_diracs( diracs, distribution, nb_diracs );

    auto time = Tick::get_time();

    // bounds
    using Bounds = PowerDiagram::Bounds::ConvexPolyhedronAssembly<Pc>;
    Bounds bounds;
    bounds.add_box( { 0, 0 }, { 1, 1 } );

    // grid
    ZGrid grid( max_pts_per_cell, 2.0 );
    grid.init( diracs );

    // computation of volumes
    VtkOutput<1> vo( { "num" } );
    std::vector<double> volumes( diracs.size(), 17 );
    grid.for_each_laguerre_cell( [&]( LC &lc, std::size_t num ) {
        if ( ! vo_output.empty() )
            lc.display( vo, { 1.0 * num } );
        volumes[ num ] = lc.measure();
    }, bounds.englobing_convex_polyhedron(), diracs );

    double t = Tick::elapsed_since( time );

    #ifdef MDPC
    std::string cmd = "cat /proc/" + std::to_string( getpid() ) + "/maps";
    system( cmd.c_str() );
    #endif

    // check volumes
    double volume = 0;
    for( double v : volumes )
        volume += v;
    P( volume, t );
    if ( ! vo_output.empty() ) {
        for( const Dirac &d : diracs )
            vo.add_point( d.pos );
        vo.save( vo_output );
    }
}

#ifndef MDPC
#define MDPC 17
#endif

int main( int argc, char **argv ) {
    // TODO: dynamic list with offsets for diracs per cell

    // max 19 par cellule => 7 en moyenne
    // max  9 par cellule => 4 en moyenne

    // sur PC fixe avec 2M: regular ( 9) => 1.64; random (11) => 2.17; lines (19) => 4.55
    //   avec TF pos, size:                 1.53                 1.95
    //
    struct Pc {
        enum { nb_bits_per_axis = 31 };
        enum { allow_ball_cut   = 0  };
        enum { dim              = 2  };
        using  TI               = std::size_t;
        using  TF               = double;
    };

    std::string vtk_output;
    if ( argc > 4 )
        vtk_output = argv[ 4 ];

    launch_bench( Pc(), argv[ 1 ], atoi( argv[ 2 ] ), atof( argv[ 3 ] ), vtk_output );
}
