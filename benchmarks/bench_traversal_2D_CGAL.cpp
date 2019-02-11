#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>
#include "../src/PowerDiagram/system/Tick.h"
#include <stdlib.h>
#include <string>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using R = CGAL::Regular_triangulation_2<K>;

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
}

int main( int, char **argv ) {
   std::vector<R::Weighted_point> diracs;
   set_up_diracs( diracs, argv[ 1 ], atof( argv[ 2 ] ) );

   auto time = Tick::get_time();
   R rt( diracs.begin(), diracs.end() );
   // rt.is_valid();

   std::string cmd = "cat /proc/" + std::to_string( getpid() ) + "/maps";
   system( cmd.c_str() );

   double s = 0;
   for( auto v = rt.all_vertices_begin(); v != rt.all_vertices_end(); ++v ) {
       auto circulator = rt.incident_faces( v ), done( circulator );
       do {
           double v = circulator->vertex( 0 )->point().point().x();
           s += v;
       } while( ++circulator != done );
   }

   std::cout << s << " " << Tick::elapsed_since( time ) << std::endl;
   return 0;

}

