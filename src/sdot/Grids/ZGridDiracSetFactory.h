#pragma once

#include "ZGridDiracSet.h"

namespace sdot {

/**
*/
class ZGridDiracSetFactory {
public:
    virtual               ~ZGridDiracSetFactory() {}
    virtual ZGridDiracSet *New                 ( ST size ) const = 0;
};

}
