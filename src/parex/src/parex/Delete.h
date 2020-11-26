#pragma once

struct Delete {
    template<class T>
    void operator()( T *data ) { delete data; }
};

