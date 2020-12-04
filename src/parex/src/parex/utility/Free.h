#pragma once

namespace parex {

struct Free {
    template<class T>
    void operator()( T *data ) { data->~T(); free( data ); }
};

} // namespace parex
