#pragma once

#include "CompiledSymbolMap.h"
class Type;

/***/
class Destructors : public CompiledSymbolMap<Type *,void(void*)> {
protected:
    virtual Path output_directory( const std::string &/*parameters*/ ) const override;

    virtual void make_srcs       ( SrcWriter &ff ) const override;
};

extern Destructors destructors;
