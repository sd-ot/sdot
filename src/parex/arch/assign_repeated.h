#pragma once

#include "SimdRange.h"
#include "SimdVec.h"

namespace parex {

template<class T,class I>
void assign_repeated( T *dst, T val, I beg, I end ) {
    SimdRange<SimdSize<T>::value>::for_each_al( beg, end, [&]( T index, auto s ) {
        SimdVec<T,s.value>::store( dst + index, val );
    } );
}

} // namespace parex
