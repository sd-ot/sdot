#include "SrcSet.h"
#include <fstream>

namespace parex {

SrcSet::SrcSet() {
    compilation_environment.include_directories << PAREX_DIR "/src";
    compilation_environment.include_directories << ASIMD_DIR "/src";
}

Src &SrcSet::src( const Path &filename ) {
    auto iter = src_map.find( filename );
    if ( iter == src_map.end() )
        iter = src_map.insert( iter, { filename, Src{ filename, compilation_environment } } );
    return iter->second;
}

void SrcSet::write_files( const Path &directory ) const {
    for( const auto &p : src_map ) {
        Path name = directory / p.second.filename();
        std::ofstream fout( name );
        p.second.write_to( fout );
    }
}

std::string SrcSet::summary() const {
    std::ostringstream ss;
    for( const auto &p : src_map ) {
        std::ostringstream ls;
        p.second.write_to( ls );

        ss << p.first << "\n";
        ss << ls.str().size() << "\n";
        ss << ls.str() << "\n";
    }
    return ss.str();
}

SrcSet::operator bool() const {
    return ! src_map.empty();
}

} // namespace parex
