#include <parex/containers/Vec.h>
#include <cstdint>
#include <limits>
#include <random>

using namespace parex;

template<class T> typename std::enable_if<std::is_floating_point<T>::value,T>::type default_max() { return - std::numeric_limits<T>::max(); }

template<class T> T default_min() { return std::numeric_limits<T>::max(); }

template<class T>
std::tuple<T *,T *> min_max( const Vec<T> &v ) {
    // TODO: simd
    T *min = new T( default_min<T>() );
    T *max = new T( default_max<T>() );
    for( const T &val : v ) {
        if ( *min > val ) *min = val;
        if ( *max < val ) *max = val;
    }

    return { min, max };
}
