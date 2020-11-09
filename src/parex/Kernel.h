#pragma once

#include "TaskRef.h"

namespace parex {

/**
*/
class Kernel {
public:
    struct                   Parameter   { std::string name; TaskRef value; };

    bool                     operator<   ( const Kernel &that ) const;

    std::string              name;                  ///<
    bool                     task_as_arg = false;   ///<
    std::vector<Parameter>   parameters  = {};      ///<
};

} // namespace parex
