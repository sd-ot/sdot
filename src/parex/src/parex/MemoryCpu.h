#pragma once

#include "Memory.h"

/**
*/
class MemoryCpu : public Memory {
public:
    virtual std::string allocator( CompilationEnvironment &compilation_environment, Type *type ) const override;
    virtual std::string name     () const override;
};

extern MemoryCpu memory_cpu;
