#pragma once

#include "KernelWithCompiledCode.h"
/**
*/
class KernelWithCodeInInclude : public KernelWithCompiledCode {
public:
    /**/                KernelWithCodeInInclude( const Path &path );

    virtual void        write_to_stream        ( std::ostream &os ) const override;

protected:
    virtual Path        output_directory       ( const std::string &parameters ) const override;
    virtual void        get_summary            ( std::ostream &os ) const override;
    virtual std::string func_name              ( const std::string &parameters ) const override;

    virtual void        make_srcs              ( SrcSet &ff ) const override;

    Path                path;
};

