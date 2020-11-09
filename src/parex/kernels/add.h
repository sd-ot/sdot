#include <parex/containers/Vec.h>
#include <parex/support/P.h>
#include <cstdint>

using namespace parex;

template<class T>
T *add( const T &a, const T &b ) {
    return new T( a + b );
}
