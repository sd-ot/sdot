#pragma once

#include <vector>

template<class TI>
std::vector<TI> range( TI end ) {
    std::vector<TI> res( end );
    for( TI i = 0; i < end; ++i )
        res[ i ] = i;
    return res;
}
