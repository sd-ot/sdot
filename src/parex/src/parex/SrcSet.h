#pragma once

#include "Src.h"
#include <map>

/**
*/
class SrcSet {
public:
    using        Path                         = Src::Path;
    using        VUPath                       = VecUnique<Path>;
    using        VUString                     = VecUnique<std::string>;
    using        MapPathSrc                   = std::map<Path,Src>;

    /**/         SrcSet                       ();

    Src&         src                          ( const Path &filename = "main.cpp" );

    void         write_files                  ( const Path &directory ) const;
    std::string  summary                      () const;
    operator     bool                         () const;

    VUPath       default_include_directories; ///< used for creation of Src
    VUString     default_cpp_flags;           ///< used for creation of Src
    VUString     default_includes;            ///< used for creation of Src
    VUString     default_prelims;             ///< used for creation of Src
    MapPathSrc   src_map;                     ///<
};

