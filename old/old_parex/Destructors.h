#pragma once

#include "CompiledSymbolMap.h"

/***/
class Destructors : public CompiledSymbolMap {
protected:
    virtual Path output_directory( const std::string &/*parameters*/ ) const override;

    virtual void make_srcs( SrcSet &ff ) const override;
};

extern Destructors destructors;
