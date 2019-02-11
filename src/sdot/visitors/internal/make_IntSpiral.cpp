#include "../system/CrossProdOfRanges.h"
#include <algorithm>
#include <cmath>

//// nsmake cpp_flag -g3

using TI = std::ptrdiff_t;

//void make_2d_points() {
//    using PT = std::array<int,2>;
//    std::vector<PT> points;

//    auto dist = [&]( PT pos ) {
//        int res = std::numeric_limits<int>::max();
//        for( std::size_t comb = 0; comb < 16; ++comb ) {
//            int dx = pos[ 0 ] + bool( comb & 1 ) - bool( comb & 4 );
//            int dy = pos[ 1 ] + bool( comb & 2 ) - bool( comb & 8 );
//            int trial = dx * dx + dy * dy;
//            res = std::min( res, trial );
//        }
//        return res;
//    };

//    auto norm = [&]( PT pos ) {
//        return pos[ 0 ] * pos[ 0 ] + pos[ 1 ] * pos[ 1 ];
//    };

//    constexpr int r = 50, n = 2 * r + 1;
//    for( int i = 0; i < n * n; ++i ) {
//        PT p{ ( i % n ) - r, ( i / n ) - r };
//        if ( dist( p ) < r * r )
//            points.push_back( p );
//    }

//    std::sort( points.begin(), points.end(), [&]( PT a, PT b ) {
//        return std::make_tuple( dist( a ), norm( a ), a[ 1 ], a[ 0 ] ) <
//               std::make_tuple( dist( b ), norm( b ), b[ 1 ], b[ 0 ] );
//    } );

//    //    constexpr int rt = r + 1;
//    //    std::array<std::array<int,2*rt+1>,2*rt+1> tab;
//    //    for( auto &v : tab )
//    //        for( auto &a : v )
//    //            a = 100;
//    //    for( int i = 0; i < points.size(); ++i )
//    //        tab[ points[ i ][ 1 ] + rt ][ points[ i ][ 0 ] + rt ] = 200 + i;
//    //    for( const auto &v : tab ) {
//    //        std::cout << "\n";
//    //        for( const auto &a : v ) {
//    //            if ( a == 100 )
//    //                std::cout << "    ";
//    //            else
//    //                std::cout << std::setw( 3 ) << a - 200 << " ";
//    //        }
//    //        std::cout << "\n";
//    //    }

//    for( size_t i = 0; i < points.size(); ++i )
//        std::cout << ( i % 8 ? "" : "\n" ) << std::setw( 3 ) << points[ i ][ 0 ] << ", " << std::setw( 3 ) << points[ i ][ 1 ] << ", " << std::setw( 4 ) << dist( points[ i ] ) << ", ";
//    std::cout << "\n";
//    std::cout << "r = " << r << "\n";
//    std::cout << "n = " << points.size() << "\n";
//}

//void make_3d_points() {
//    using PT = std::array<int,3>;
//    std::vector<PT> points;

//    auto dist = [&]( PT pos ) {
//        int res = std::numeric_limits<int>::max();
//        for( std::size_t comb = 0; comb < 4 * 4 * 4; ++comb ) {
//            int dx = pos[ 0 ] + bool( comb & 1 ) - bool( comb &  8 );
//            int dy = pos[ 1 ] + bool( comb & 2 ) - bool( comb & 16 );
//            int dz = pos[ 2 ] + bool( comb & 4 ) - bool( comb & 32 );
//            int trial = dx * dx + dy * dy + dz * dz;
//            res = std::min( res, trial );
//        }
//        return res;
//    };

//    auto norm = [&]( PT pos ) {
//        return pos[ 0 ] * pos[ 0 ] + pos[ 1 ] * pos[ 1 ] + pos[ 2 ] * pos[ 2 ];
//    };

//    constexpr int r = 25, n = 2 * r + 1;
//    for( int i = 0; i < n * n * n; ++i ) {
//        PT p{ i % n - r, ( i / n ) % n - r, i / n / n - r };
//        if ( dist( p ) < r * r )
//            points.push_back( p );
//    }

//    std::sort( points.begin(), points.end(), [&]( PT a, PT b ) {
//        return std::make_tuple( dist( a ), norm( a ), a[ 2 ], a[ 1 ], a[ 0 ] ) <
//               std::make_tuple( dist( b ), norm( b ), b[ 2 ], b[ 1 ], b[ 0 ] );
//    } );

//    //    constexpr int rt = r + 1;
//    //    std::array<std::array<std::array<int,2*rt+1>,2*rt+1>,2*rt+1> tab;
//    //    for( auto &v : tab )
//    //        for( auto &b : v )
//    //            for( auto &a : b )
//    //                a = 100;
//    //    for( int i = 0; i < points.size(); ++i )
//    //        tab[ points[ i ][ 2 ] + rt ][ points[ i ][ 1 ] + rt ][ points[ i ][ 0 ] + rt ] = 200 + i;
//    //    for( const auto &v : tab ) {
//    //        for( const auto &b : v ) {
//    //            std::cout << "\n";
//    //            for( const auto &a : b ) {
//    //                if ( a == 100 )
//    //                    std::cout << "    ";
//    //                else
//    //                    std::cout << std::setw( 3 ) << a - 200 << " ";
//    //            }
//    //            std::cout << "\n";
//    //        }
//    //    }

//    for( size_t i = 0; i < points.size(); ++i )
//        std::cout << ( i % 8 ? "" : "\n" ) << std::setw( 3 ) << points[ i ][ 0 ] << ", " << std::setw( 3 ) << points[ i ][ 1 ] << ", " << std::setw( 3 ) << points[ i ][ 2 ] << ", " << std::setw( 4 ) << dist( points[ i ] ) << ", ";
//    std::cout << "\n";
//    std::cout << "r = " << r << "\n";
//    std::cout << "n = " << points.size() << "\n";
//}

template<int dim>
void make( TI l ) {
    using std::pow;
    using std::min;

    std::array<TI,dim> beg, end;
    for( size_t d = 0; d < dim; ++d ) {
        beg[ d ] = - l;
        end[ d ] = l + 1;
    }
    std::vector<std::array<TI,dim>> points;
    CrossProdOfRanges<TI,dim> cpr( beg, end );
    cpr.for_each( [&]( auto p ) {
        TI n2 = 0;
        for( size_t d = 0; d < dim; ++d )
            n2 += p[ d ] * p[ d ];
        if ( n2 && n2 <= l * l )
            points.push_back( p );
    } );

    auto dist = []( std::array<TI,dim> a ) {
        TI res = 0;
        for( size_t d = 0; d < dim; ++d )
            res += pow( a[ d ] > 0 ? a[ d ] - 1 : ( a[ d ] < 0 ? a[ d ] + 1 : 0 ), 2 );
        return res;
    };
    std::sort( points.begin(), points.end(), [&]( std::array<TI,dim> a, std::array<TI,dim> b ) {
        return dist( a ) < dist( b );
    } );

    std::cout << "const std::size_t nb_values_" << dim << "D = " << points.size() << ";\n";
    std::cout << "const std::ptrdiff_t values_" << dim << "D[] = {";
    for( size_t i = 0; i < points.size(); ++i ) {
        std::cout << ( i % 8 == 0 ? "\n    " : " " );
        std::cout << std::setw( 5 ) << dist( points[ i ] ) << ",";
        for( size_t d = 0; d < dim; ++d )
            std::cout << std::setw( 5 ) << points[ i ][ d ] << ",";
    }
    std::cout << "\n};\n";
}

int main() {
    make<2>( 50 );
//    make<3>( 50 );
}
