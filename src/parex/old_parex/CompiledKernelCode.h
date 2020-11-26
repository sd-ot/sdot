#pragma once

#include "CompiledSymbolMap.h"
class KernelWithCompiledCode;
class TypeInfo;
class Type;
class Task;

/***/
class CompiledKernelCode : public CompiledSymbolMap {
public:
    using                   Func              = void( Task * );

    /**/                    CompiledKernelCode( KernelWithCompiledCode *kernel );
    Func*                   get_func          ( const Task *task );

protected:
    virtual Path            output_directory  ( const std::string &parameters ) const override;
    virtual void            make_srcs         ( SrcWriter &ff ) const override;

    KernelWithCompiledCode* kernel;
};

extern CompiledKernelCode compiled_kernels;
