#pragma once

#include <ostream>

namespace sdot {

template<class T>
struct BinaryRepr {
    const T &val;
};

template<class T>
std::ostream &operator<<( std::ostream &os, const BinaryRepr<T> &val ) {
    for( size_t i = 8 * sizeof( val.val ); i--; )
        os << ( ( val.val >> i ) & 1 ? '1' : '0' );
    return os;
}

template<class T>
BinaryRepr<T> binary_repr( const T &val ) {
    return { val };
}

}
