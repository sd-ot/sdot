#include "CompiledSymbolMap.h"
#include "SrcWriter.h"
#include "TmpDir.h"
#include "TODO.h"

Src &SrcWriter::src( std::string filename ) {
    // try without _number.ext
    auto iter = src_map.find( filename );
    if ( iter == src_map.end() )
        iter = src_map.insert( iter, { filename, Src{ tmp_dir->p / filename, default_cpp_flags, default_includes, default_include_directories } } );
    return iter->second;
}

void SrcWriter::close() {
    for( auto &p : src_map )
        p.second.fout.close();
}

std::string SrcWriter::main_cpp_name() const {
    return compiled_symbol_map->symbol_name( parameters ) + ".cpp";
}
