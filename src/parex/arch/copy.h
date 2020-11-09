#pragma once

#include "SimdRange.h"
#include "SimdVec.h"

namespace parex {

template<class T,class TI>
void copy( T *dst, const T *src, TI len ) {
    SimdRange<SimdSize<T>::value>::for_each( len, [&]( TI index, auto s ) {
        using SV = SimdVec<T,s.value>;
        SV::store_aligned( dst + index, SV::load_aligned( src + index ) );
    } );
}

} // namespace parex
