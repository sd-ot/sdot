#include <parex/containers/Vec.h>
#include <cstdint>
#include <random>

using namespace parex;

template<class T>
Vec<T> *random_vec( std::size_t size, T min, T max ) {
    std::uniform_real_distribution<T> dis( min, max );
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

    Vec<T> *res = new Vec<T>( size );
    for( T &val : *res )
        val = dis( gen );

    return res;
}

