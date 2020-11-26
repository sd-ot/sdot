#include "CompiledSymbolMap.h"
#include "SrcWriter.h"
#include "TODO.h"

Src &SrcWriter::src( const Path &filename ) {
    auto iter = src_map.find( filename );
    if ( iter == src_map.end() )
        iter = src_map.insert( iter, { filename, Src{ default_include_directories, default_cpp_flags, default_includes, default_prelims } } );
    return iter->second;
}

