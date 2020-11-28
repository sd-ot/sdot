#pragma once

#include "../GeneratedLibrarySet.h"
class Type;

/**
*/
class CompiledTaskLib : public GeneratedLibrarySet {
public:
    /***/               CompiledTaskLib( const Path &src_path, const std::vector<Type *> &children_types );

protected:
    virtual void        make_srcs      ( SrcSet &sw ) override;
    static std::string  summary        ( const Path &src_path, const std::vector<Type *> &children_types );

    std::vector<Type *> children_types;
    Path                src_path;
};

