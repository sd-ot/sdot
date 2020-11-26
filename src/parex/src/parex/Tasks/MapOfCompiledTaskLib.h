#pragma once

#include "CompiledTaskLib.h"

/**
*/
class MapOfCompiledTaskLib {
public:
    using            Path              = GeneratedLib::Path;

    CompiledTaskLib* lib               ( const Path &src_path, const std::vector<Type *> &children_types );

private:
    using            Params            = std::tuple<Path,std::vector<Type *>>;
    using            Map               = std::map<Params,CompiledTaskLib>;

    Map              map;              ///<
};

