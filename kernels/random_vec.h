#include "../src/sdot/support/P.h"
#include <cstdint>
#include <random>

template<class T>
T *random_vec( std::size_t size, T min, T max ) {
    //    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    //    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    //    std::uniform_real_distribution<> dis(1.0, 2.0);
    return new T( 180 );
}
