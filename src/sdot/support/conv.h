#pragma once

#include <parex/utility/S.h>

namespace sdot {

template<class T,class G>
G conv( const T &val, parex::S<G> ) {
    return G( val );
}

}
