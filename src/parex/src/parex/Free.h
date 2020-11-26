#pragma once

struct Free {
    template<class T>
    void operator()( T *data ) { data->~T(); free( data ); }
};
