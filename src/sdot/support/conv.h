#pragma once

#include "S.h"

template<class T,class G>
G conv( const T &val, S<G> ) {
    return G( val );
}
