#include "SrcSet.h"
#include <fstream>

Src &SrcSet::src( const Path &filename ) {
    auto iter = src_map.find( filename );
    if ( iter == src_map.end() )
        iter = src_map.insert( iter, { filename, Src{ default_include_directories, default_cpp_flags, default_includes, default_prelims } } );
    return iter->second;
}

void SrcSet::write_files( const Path &directory ) const {
    for( const auto &p : src_map ) {
        std::ofstream fout( directory / p.first );
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

