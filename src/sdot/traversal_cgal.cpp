//// nsmake avoid_inc CGAL/
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>
#include "traversal_cgal.h"

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O5
//// nsmake lib_name CGAL
//// nsmake lib_name gmp

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using R = CGAL::Regular_triangulation_2<K>;

using Segment = K::Segment_2;
using Ray = K::Ray_2;

using std::sqrt;
using std::pow;

double traversal_cgal( const double *positions, const double *weights, int nb_diracs ) {
    std::vector<R::Weighted_point> diracs;
    for( int i = 0; i < nb_diracs; ++i )
        diracs.push_back( { { positions[ 2 * i + 0 ], positions[ 2 * i + 1 ] }, weights[ i ] } );

    R rt( diracs.begin(), diracs.end() );
    // rt.is_valid();

    double s = 0;
    for( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
        auto c = rt.incident_edges (v), d(c);
        do {
            if ( rt.is_infinite( c ) )
              continue;

            CGAL::Object o = rt.dual( *c );
            auto smurf = ( c->first )->vertex( rt.ccw( c->second ) ); // other vertex
            if ( const Segment *pseg = CGAL::object_cast<Segment>( &o ) ) {
                auto a = pseg->source();
                auto b = pseg->target();
                s += sqrt( pow( a.x() - b.x(), 2 ) + pow( a.y() - b.y(), 2 ) );
            } else if ( const Ray *pray = CGAL::object_cast<Ray>( &o ) ) {
            }
        } while (++c != d);
    }

    return s;
}
