#pragma once

#include <parex/support/S.h>

namespace sdot {

template<class T,class G>
G conv( const T &val, parex::S<G> ) {
    return G( val );
}

}
