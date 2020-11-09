#pragma once

#include "SimdRange.h"
#include "SimdVec.h"

namespace parex {

template<class T>
void assign_iota( T *data, T beg, T end, T ini = 0 ) {
    SimdRange<SimdSize<T>::value>::for_each_with_iota( beg, end, [&]( T index, auto iota, auto s ) {
        SimdVec<T,s.value>::store( data + index, iota ); // _aligned
    }, ini );
}

} // namespace parex
