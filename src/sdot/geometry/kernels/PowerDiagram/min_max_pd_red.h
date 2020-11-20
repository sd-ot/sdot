#include <algorithm>
using std::min;
using std::min;
using std::max;

template<class TA>
TA *min_max_pd_red( const TA &a, const TA &b ) {
    TA *res = new TA( a.size() );
    for( std::size_t i = 0; i < a.size(); ++i )
        res->operator[]( i ) = {
            min( a[ i ][ 0 ], b[ i ][ 0 ] ),
            max( a[ i ][ 1 ], b[ i ][ 1 ] )
        };
    return res;
}

template<class TA>
TA *min_max_pd_red( const TA &a, const TA &b, const TA &c ) {
    TA *res = new TA( a.size() );
    for( std::size_t i = 0; i < a.size(); ++i )
        res->operator[]( i ) = {
            min( min( a[ i ][ 0 ], b[ i ][ 0 ] ), c[ i ][ 0 ] ),
            max( max( a[ i ][ 1 ], b[ i ][ 1 ] ), c[ i ][ 1 ] )
        };
    return res;
}

template<class TA>
TA *min_max_pd_red( const TA &a, const TA &b, const TA &c, const TA &d ) {
    TA *res = new TA( a.size() );
    for( std::size_t i = 0; i < a.size(); ++i )
        res->operator[]( i ) = {
            min( min( a[ i ][ 0 ], b[ i ][ 0 ] ), min( c[ i ][ 0 ], d[ i ][ 0 ] ) ),
            max( max( a[ i ][ 1 ], b[ i ][ 1 ] ), max( c[ i ][ 1 ], d[ i ][ 1 ] ) )
        };
    return res;
}
