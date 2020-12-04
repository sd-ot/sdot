#pragma once

namespace parex {

struct Delete {
    template<class T>
    void operator()( T *data ) { delete data; }
};

} // namespace parex
