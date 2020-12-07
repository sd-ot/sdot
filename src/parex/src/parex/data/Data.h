#pragma once

#include "Type.h"

namespace parex {

/**
*/
class Data {
public:
    /**/  Data();

    void* data;
    Type* type;
    bool  own;
};

} // namespace parex
