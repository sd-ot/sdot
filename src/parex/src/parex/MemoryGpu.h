#pragma once

#include "Memory.h"
class Src;

/**
*/
class MemoryGpu : public Memory {
public:
    /**/                MemoryGpu( int num_gpu );

    virtual std::string allocator( CompilationEnvironment &compilation_environment, Type *type ) const override;
    virtual std::string name     () const override;

    int                 num_gpu = 0;
};

