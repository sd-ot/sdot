#pragma once

#include "SimdRange.h"
#include "SimdVec.h"

namespace parex {

template<class T,class I,class G>
void assign_repeated( T *dst, I beg, I end, const G &val ) {
    SimdRange<SimdSize<T>::value>::for_each_al( beg, end, [&]( I index, auto s ) {
        SimdVec<T,s.value>::store( dst + index, val );
    } );
}

} // namespace parex
