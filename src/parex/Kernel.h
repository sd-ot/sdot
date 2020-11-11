#pragma once

#include "TaskRef.h"

namespace parex {

/**
*/
class Kernel {
public:
    bool        operator<       ( const Kernel &that ) const;

    std::string name;                      ///<
    bool        task_as_arg     = false;   ///<
};

} // namespace parex
