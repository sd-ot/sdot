//// nsmake avoid_inc CGAL/
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>
#include "../src/sdot/system/Time.h"
#include "set_up_diracs.h"
#include <cxxopts.hpp>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using R = CGAL::Regular_triangulation_2<K>;

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O5
//// nsmake lib_flag -O5

//// nsmake lib_name CGAL
//// nsmake lib_name gmp

int main( int argc, char **argv ) {
    struct Pc { enum { nb_bits_per_axis = 31, allow_ball_cut = 0, dim = 2 }; using TI = std::size_t; using TF = double; };
    using  Pt = Point2<Pc::TF>;
    using  TF = Pc::TF;

    // options
    cxxopts::Options options( argv[ 0 ], "bench volume");
    options.add_options()
        ( "d,distribution"          , "distribution name (regular, random, ...)", cxxopts::value<std::string>()->default_value( "regular" ) )
        ( "n,nb-diracs"             , "...", cxxopts::value<double>()->default_value( "100" ) )
        ( "periodic"                , "" )
        ( "h,help"                  , "get help" )
        ;
    auto args = options.parse( argc, argv );

    // diracs
    std::vector<TF> weights;
    std::vector<Pt> positions;
    set_up_diracs( positions, weights, args[ "distribution" ].as<std::string>(), args[ "nb-diracs" ].as<double>() );

    std::vector<R::Weighted_point> diracs( weights.size() );
    for( size_t i = 0; i < weights.size(); ++i )
        diracs[ i ] = { { positions[ i ].x, positions[ i ].y }, weights[ i ] };

    auto t0 = Time::get_time();
    R rt( diracs.begin(), diracs.end() );

    auto t1 = Time::get_time();
    double s = 0;
    for( auto v = rt.all_vertices_begin(); v != rt.all_vertices_end(); ++v ) {
        auto circulator = rt.incident_faces( v ), done( circulator );
        do {
            double v = circulator->vertex( 0 )->point().point().x();
            s += v;
        } while( ++circulator != done );
    }
    auto t2 = Time::get_time();

    P( Time::delta( t0, t1 ) );
    P( Time::delta( t1, t2 ) );
    P( Time::delta( t0, t2 ) );
}
