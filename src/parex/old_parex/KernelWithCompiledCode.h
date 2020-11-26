#pragma once

#include "CompiledKernelCode.h"
#include "Kernel.h"

/**
*/
class KernelWithCompiledCode : public Kernel {
public:
    using                      Path                  = CompiledKernelCode::Path;

    /**/                       KernelWithCompiledCode();

    virtual void               exec                  ( Task *task ) const;

protected:
    friend class               CompiledKernelCode;

    std::string                kernel_parameters     ( const Task *task );
    std::vector<TypeInfo *>    type_infos            ( const std::string &kernel_parameters ) const;

    virtual Path               output_directory      ( const std::string &parameters ) const = 0;
    virtual void               get_summary           ( std::ostream &os ) const = 0;
    virtual std::string        func_name             ( const std::string &parameters ) const = 0;

    virtual void               make_srcs             ( SrcWriter &ff ) const;

    mutable CompiledKernelCode ckc;                  ///< compiled code
};

