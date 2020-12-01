#pragma once

#include "../SimdRange.h"
#include "../SimdVec.h"

namespace asimd {

template<class T,class U,class V>
void assign_iota( T *data, U beg, V len ) {
    SimdRange<SimdSize<T>::value>::template for_each_with_iota<T>( 0, len, [&]( T index, auto iota, auto s ) {
        SimdVec<T,s.value>::store( data + index, iota ); // _aligned
    }, beg );
}

} // namespace asimd
