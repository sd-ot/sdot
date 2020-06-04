#pragma once

#include <cmath>

namespace sdot {

template<class T,class U> constexpr
T ceil( T a, U m ) {
    if ( not m )
        return a;
    return ( a + m - 1 ) / m * m;
}

template<class T,class U>
T gcd( T a, U b ) {
    if ( b == 1 )
        return 1;

    T old;
    while ( b ) {
        old = b;
        b = a % b;
        a = old;
    }
    return a;
}

template<class T,class U>
T lcm( T a, U b ) {
    return a * b / gcd( a, b );
}

template<class T>
T pow_2_le( T a ) {
    for( T r = 1; ; r *= 2 )
        if ( r >= a )
            return r;
}

}
