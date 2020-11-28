#pragma once

#include "Src.h"
#include <map>

/**
*/
class SrcWriter {
public:
    using            Path                         = Src::Path;
    using            String                       = std::string;
    using            MapStringSrc                 = std::map<String,Src>;
    using            VecUniqueString              = VecUnique<String>;

    Src&             src                          ( const Path &filename );

    VecUniqueString  default_include_directories; ///<
    VecUniqueString  default_cpp_flags;           ///<
    VecUniqueString  default_includes;            ///<
    VecUniqueString  default_prelims;             ///<
    MapStringSrc     src_map;                     ///<
};

