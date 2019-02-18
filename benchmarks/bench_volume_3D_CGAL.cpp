//// nsmake avoid_inc CGAL/
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Periodic_3_regular_triangulation_3.h>
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
    using  Pt = sdot::Point3<TF>;

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
    for( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
        std::list<R::Edge> edges;
        rt.incident_edges( v, std::back_inserter( edges ) );

        TF vol( 0 );
        for(typename std::list<R::Edge>::iterator eit = edges.begin(); eit != edges.end(); ++eit) {
          // compute the dual of the edge *eit but handle the translations
          // with respect to the dual of v. That is why we cannot use one
          // of the existing dual functions here.
          R::Facet_circulator fstart = rt.incident_facets( *eit );
          R::Facet_circulator fcit = fstart;
          std::vector<R::Point_3> pts;
          do {
            // TODO: possible speed-up by caching the circumcenters
            R::Point_3 dual_orig = fcit->first->weighted_circumcenter();
            int idx = fcit->first->index(v);
            int off = idx;
            pts.push_back( dual_orig );
            ++fcit;
          } while(fcit != fstart);

          R::Point_3 orig(0,0,0);
          for(unsigned int i=1; i<pts.size()-1; i++)
            vol += R::Tetrahedron(orig,pts[0],pts[i],pts[i+1]).volume();
        }
        s += vol;


        //        s += rt.dual_volume();
        //        auto circulator = rt.incident_facets( *v ), done( circulator );
        //        do {
        //            double v = circulator->first->vertex( 0 )->point().point().x();
        //            s += v;
        //        } while( ++circulator != done );
    }
    auto t2 = Time::get_time();

    P( Time::delta( t0, t1 ) );
    P( Time::delta( t1, t2 ) );
    P( Time::delta( t0, t2 ) );
}
