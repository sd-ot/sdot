#include <algorithm>
#include "Src.h"

Src::Src( const Path &filename, std::string default_cpp_flags, std::vector<std::string> default_includes, std::vector<std::string> default_include_directories ) : fout( filename ) {
    add_cpp_flags( default_cpp_flags );

    for( const auto &include : default_includes )
        add_include( include );

    for( const auto &include_directory : default_include_directories )
        add_include_directory( include_directory );
}

void Src::add_cpp_flags( std::string flags ) {
    if ( ! cpp_flags.empty() )
        cpp_flags += " ";
    cpp_flags += flags;
}

void Src::add_include( std::string include ) {
    if ( std::find( includes.begin(), includes.end(), include ) == includes.end() )
        includes.push_back( include );
}

void Src::add_include_directory( std::string include_directory ) {
    if ( std::find( include_directories.begin(), include_directories.end(), include_directory ) == include_directories.end() )
        include_directories.push_back( include_directory );
}
