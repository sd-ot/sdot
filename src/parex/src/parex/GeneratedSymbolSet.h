#pragma once

#include "GeneratedLibrarySet.h"
#include <unordered_map>

/**

*/
class GeneratedSymbolSet {
public:
    using                Path              = GeneratedLibrarySet::Path;

    /**/                 GeneratedSymbolSet( const Path &output_directory = ".generated_libs" ) : lib_set( output_directory ) {}
    template<class T> T* get_symbol        ( const std::function<void(SrcSet&)> &src_writer, std::string summary_of_parameters = {}, const std::string &symbol_name = "exported", SrcSet &&src_set = {} );

private:
    using                SymbolMap         = std::unordered_map<std::string,void *>;

    SymbolMap            symbol_map;       ///<
    GeneratedLibrarySet  lib_set;          ///<
};

// ---------------------------------------------------------------------------
template<class T> T *GeneratedSymbolSet::get_symbol( const std::function<void(SrcSet&)> &src_writer, std::string summary_of_parameters, const std::string &symbol_name, SrcSet &&src_set ) {
    // if need to make a summary, use the sources
    if ( summary_of_parameters.empty() ) {
        if ( ! src_set )
            src_writer( src_set );
        summary_of_parameters = src_set.summary();
    }

    // make lib and symbol if not already done
    std::string key = symbol_name + "\n" + summary_of_parameters;
    auto iter = symbol_map.find( key );
    if ( iter == symbol_map.end() )
        iter = symbol_map.insert( iter, { key, lib_set.get_library( src_writer, summary_of_parameters, std::move( src_set ) )->symbol<void *>( symbol_name ) } );

    //
    return reinterpret_cast<T *>( iter->second );
}
