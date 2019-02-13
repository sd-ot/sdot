#pragma once

#include "../src/sdot/system/Assert.h"
#include "../src/sdot/system/Mpi.h"
#include "../src/sdot/Point3.h"
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include <cmath>

template<class Pt,class TF>
void set_up_diracs( std::vector<Pt> &positions, std::vector<TF> &weights, std::string distribution, std::size_t nb_diracs ) {
    using TI = std::size_t;
    using std::sqrt;
    using std::pow;
    using std::max;
    using std::min;

    if ( distribution == "regular" ) {
        TI l = pow( nb_diracs, 1.0 / 3.0 );
        positions.resize( l * l * l );
        weights.resize( l * l * l );
        for( TI i = 0, c = 0; i < l; ++i ) {
            for( TI j = 0; j < l; ++j ) {
                for( TI k = 0; k < l; ++k, ++c ) {
                    positions[ c ] = {
                        ( k + 0.45 + 0.1 * rand() / RAND_MAX ) / l,
                        ( j + 0.45 + 0.1 * rand() / RAND_MAX ) / l,
                        ( i + 0.45 + 0.1 * rand() / RAND_MAX ) / l
                    };
                    weights[ c ] = 1.0;
                }
            }
        }
        return;
    }

    if ( distribution == "random" ) {
        std::size_t n = nb_diracs / mpi->size() + 10 * mpi->rank();
        positions.resize( n );
        weights.resize( n );
        TF s = TF( 1 ) / mpi->size();
        TF b = s * mpi->rank();
        for( TI i = 0; i < mpi->rank(); ++i )
            rand();
        for( TI i = 0; i < n; ++i ) {
            positions[ i ] = { b + s * rand() / RAND_MAX, 1.0 * rand() / RAND_MAX, 1.0 * rand() / RAND_MAX };
            weights[ i ] = 1.0;
        }
        return;
    }

    //    if ( distribution == "gaussian" ) {
    //        positions.resize( 0 );
    //        weights.resize( 0 );

    //        std::default_random_engine generator;
    //        std::normal_distribution<double> distribution( 0.5, 0.1 );

    //        while ( weights.size() < nb_diracs ) {
    //            double x = distribution( generator );
    //            double y = distribution( generator );
    //            if ( x > 0 && x < 1 && y > 0 && y < 1 ) {
    //                positions.push_back( { x, y } );
    //                weights.push_back( 1.0 );
    //            }
    //        }
    //        return;
    //    }

    //    if ( distribution == "split" ) {
    //        positions.resize( nb_diracs );
    //        weights.resize( nb_diracs );
    //        for( std::size_t i = 0; i < nb_diracs; ++i ) {
    //            double x = 0.5 * rand() / RAND_MAX;
    //            double y = 1.0 * rand() / RAND_MAX;
    //            positions[ i ] = { x + 0.5 * ( x > 0.25 ), y };
    //            weights[ i ] = 1.0;
    //        }
    //        return;
    //    }

    //    if ( distribution == "lines" ) {
    //        positions.resize( 0 );
    //        weights.resize( 0 );
    //        for( std::size_t i = 0; i < nb_diracs / 2; ++i ) {
    //            double y = 1.0 * rand() / RAND_MAX;
    //            double x = 0.1 + 0.05 * rand() / RAND_MAX;
    //            double w = 1; // 0.5 + 0.05 * rand() / RAND_MAX;
    //            positions.push_back( { x, y } );
    //            weights.push_back( w );
    //        }
    //        for( std::size_t i = 0; i < nb_diracs / 2; ++i ) {
    //            double y = 1.0 * rand() / RAND_MAX;
    //            double x = 0.9 - 0.5 * y + 0.05 * rand() / RAND_MAX;
    //            double w = 1; // 0.5 + 0.05 * rand() / RAND_MAX;
    //            positions.push_back( { x, y } );
    //            weights.push_back( w );
    //        }

    //        return;
    //    }

    //    if ( distribution == "lines_reg" ) {
    //        positions.resize( 0 );
    //        weights.resize( 0 );
    //        for( std::size_t i = 0; i < nb_diracs / 2; ++i ) {
    //            double y = 1.0 * ( i + 0.5 ) / nb_diracs;
    //            double x = 0.1 + 1e-3 * rand() / RAND_MAX;
    //            double w = 1;
    //            positions.push_back( { x, y } );
    //            weights.push_back( w );
    //        }
    //        for( std::size_t i = 0; i < nb_diracs / 2; ++i ) {
    //            double y = 1.0 * ( i + 0.5 ) / nb_diracs;
    //            double x = 0.9 - 0.5 * y + 1e-3 * rand() / RAND_MAX;
    //            double w = 1;
    //            positions.push_back( { x, y } );
    //            weights.push_back( w );
    //        }

    //        return;
    //    }

    //    if ( distribution == "concentration" ) {
    //        positions.resize( 0 );
    //        weights.resize( 0 );
    //        //
    //        for( std::size_t i = 0; i < nb_diracs / 20; ++i ) {
    //            double y = 1.0 * rand() / RAND_MAX;
    //            double x = 1.0 * rand() / RAND_MAX;
    //            double w = 1;
    //            positions.push_back( { x, y } );
    //            weights.push_back( w );
    //        }
    //        //
    //        for( std::size_t i = 0; i < 19 * nb_diracs / 20; ++i ) {
    //            double y = 0.7 + 0.1 * rand() / RAND_MAX;
    //            double x = 0.7 + 0.1 * rand() / RAND_MAX;
    //            double w = 1; // 0.5 + 0.05 * rand() / RAND_MAX;
    //            positions.push_back( { x, y } );
    //            weights.push_back( w );
    //        }

    //        return;
    //    }

    //    if ( distribution == "2:100" ) {
    //        positions.resize( 0 );
    //        weights.resize( 0 );

    //        for( double j = 0; j <= 1; j += 1 ) {
    //            for( double i = 0; i <= 1; i += 1 ) {
    //                positions.push_back( { i, j } );
    //                weights.push_back( 1 + 0.5 * ( i == 0 && j == 0 ) );
    //            }
    //        }
    //        P( weights );

    //        return;
    //    }

    //    if ( distribution.size() > 4 && distribution.substr( distribution.size() - 4 ) == ".xyz" ) {
    //        std::ifstream f( distribution.c_str() );
    //        positions.resize( 0 );
    //        weights.resize( 0 );
    //        double x, y, z;
    //        while ( f >> x >> y >> z ) {
    //            if ( z > 0.1 && z < 0.2 ) {
    //                positions.push_back( { x, y } );
    //                weights.push_back( 1 );
    //            }
    //        }

    //        return;
    //    }

    if ( distribution.size() > 4 && distribution.substr( distribution.size() - 4 ) == ".xyzw" ) {
        std::ifstream f( distribution.c_str() );
        positions.resize( 0 );
        weights.resize( 0 );
        double x, y, z, w;
        while ( f >> x >> y >> z >> w ) {
            positions.push_back( { x, y, z } );
            weights.push_back( w );
        }

        return;
    }

    if ( distribution.size() > 6 && distribution.substr( 0, 6 ) == "eur://" ) {
        std::string filename = distribution.substr( 6 );
        std::ifstream f( filename.c_str() );
        double v[ 4 ];

        weights.resize( 0 );
        positions.resize( 0 );
        f.seekg( 0, std::ios_base::beg );
        while ( f.readsome( (char *)v, 4 * sizeof( double ) ) ) {
            positions.push_back( { v[ 0 ], v[ 1 ], v[ 2 ] } );
            weights.push_back( 1 );
        }
        return;
    }

    TODO;
}

