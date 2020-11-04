#pragma once

#include "Value.h"

namespace parex {

/**
*/
class Kernel {
public:
    struct                 Parameter  { std::string name; Value value; };

    bool                   operator<  ( const Kernel &that ) const;

    std::string            name;
    std::vector<Parameter> parameters = {};
    std::string            func = "kernel";
};

} // namespace parex
