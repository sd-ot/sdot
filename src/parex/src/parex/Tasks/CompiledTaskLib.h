#pragma once

#include "../GeneratedLib.h"
class Type;

/**
*/
class CompiledTaskLib : public GeneratedLib {
public:
    /***/               CompiledTaskLib( const Path &src_path, const std::vector<Type *> &children_types );

protected:
    virtual void        make_srcs      ( SrcWriter &sw ) override;
    static std::string  summary        ( const Path &src_path, const std::vector<Type *> &children_types );

    std::vector<Type *> children_types;
    Path                src_path;
};

