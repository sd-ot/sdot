#pragma once

#include "ZGridDiracSet.h"
#include <cstdint>

namespace sdot {

/**
*/
template<class T,class S>
class ZGridDiracSetFactory {
public:
    virtual                    ~ZGridDiracSetFactory() {}
    virtual S                   nb_diracs_for_mem   ( std::size_t mem ) const = 0;
    virtual ZGridDiracSet<T,S> *New                 ( S size ) const = 0;
};

}
