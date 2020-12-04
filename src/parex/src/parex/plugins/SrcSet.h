#pragma once

#include "Src.h"
#include <map>

namespace parex {

/**
*/
class SrcSet {
public:
    using                  Path                         = Src::Path;
    using                  MapPathSrc                   = std::map<Path,Src>;

    /**/                   SrcSet                       ();

    Src&                   src                          ( const Path &filename = "main.cpp" );

    void                   write_files                  ( const Path &directory ) const;
    std::string            summary                      () const;
    operator               bool                         () const;

    CompilationEnvironment compilation_environment;     ///<
    MapPathSrc             src_map;                     ///<
};

} // namespace parex
