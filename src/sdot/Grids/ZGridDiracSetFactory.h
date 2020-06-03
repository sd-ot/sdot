#pragma once

#include "ZGridDiracSet.h"

namespace sdot {

/**
*/
template<class T,class S>
class ZGridDiracSetFactory {
public:
    virtual                    ~ZGridDiracSetFactory() {}
    virtual ZGridDiracSet<T,S> *New                 ( S size ) const = 0;
};

}
