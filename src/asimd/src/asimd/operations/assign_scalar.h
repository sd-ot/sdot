#pragma once

#include "../SimdRange.h"
#include "../SimdVec.h"

namespace asimd {

template<class T,class G,class I>
void assign_scalar( T *dst, const G &val, I len ) {
    SimdRange<SimdSize<T>::value>::for_each( len, [&]( I index, auto s ) {
        SimdVec<T,s.value>::store( dst + index, val );
    } );
}

} // namespace asimd
