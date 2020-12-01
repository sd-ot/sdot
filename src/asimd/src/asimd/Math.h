#pragma once

namespace asimd {

template<class T,class U>
T div_up( T a, U m ) {
    return ( a + m - 1 ) / m;
}

} // namespace asimd
