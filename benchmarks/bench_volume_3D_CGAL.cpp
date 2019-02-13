//// nsmake avoid_inc CGAL/
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_3.h>
#include "../src/sdot/system/Time.h"
#include "set_up_diracs_3D.h"
#include <cxxopts.hpp>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using R = CGAL::Regular_triangulation_3<K>;

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O5
//// nsmake lib_flag -O5

//// nsmake lib_name CGAL
//// nsmake lib_name gmp

int main( int argc, char **argv ) {
    using  TF = double;
    using  Pt = Point3<TF>;

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
        diracs[ i ] = { { positions[ i ].x, positions[ i ].y, positions[ i ].z }, weights[ i ] };

    auto t0 = Time::get_time();
    R rt( diracs.begin(), diracs.end() );

    auto t1 = Time::get_time();
    double s = 0;
    for( auto v = rt.all_edges_begin(); v != rt.all_edges_end(); ++v ) {
        auto circulator = rt.incident_facets( *v ), done( circulator );
        do {
            double v = circulator->first->vertex( 0 )->point().point().x();
            s += v;
        } while( ++circulator != done );
    }
    auto t2 = Time::get_time();

    P( Time::delta( t0, t1 ) );
    P( Time::delta( t1, t2 ) );
    P( Time::delta( t0, t2 ) );
}
