#pragma once

#include "Type.h"

class CompilationEnvironment;

/**/
class Memory {
public:
    virtual            ~Memory   ();

    virtual std::string allocator( CompilationEnvironment &compilation_environment, Type *type ) const = 0;
    virtual std::string name     () const = 0;
};

