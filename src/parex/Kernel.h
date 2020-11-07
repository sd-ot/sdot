#pragma once

#include "Value.h"

namespace parex {

/**
*/
class Kernel {
public:
    struct                   Parameter  { std::string name; Value value; };

    bool                     operator<  ( const Kernel &that ) const;

    std::string              name;                  ///<
    std::vector<std::size_t> modified   = {};       ///< list of modified inputs
    std::vector<Parameter>   parameters = {};       ///<
};

} // namespace parex
