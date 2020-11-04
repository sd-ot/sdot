#pragma once

#include "Kernel.h"
#include "Value.h"

namespace parex {

class Task {
public:
    /**/                Task();

    Kernel              kernel;
    std::vector<Value>  inputs;
    mutable std::size_t cpt_use;
};

} // namespace parex
