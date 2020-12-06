#pragma once

#include "Type.h"

namespace parex {

/**
*/
class Data {
public:
    void* data;
    Type* type;
    bool  own;
};

} // namespace parex
