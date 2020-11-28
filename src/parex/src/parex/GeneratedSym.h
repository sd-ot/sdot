#pragma once

#include "GeneratedLibrarySet.h"

/**/
template<class T>
class GeneratedSym {
public:
    bool         need_init() const;
    void         init     ( const std::string &name, const std::string &summary_of_parameters, const std::function<void( SrcWriter &sw )> &src_writer, const GeneratedLibrarySet::Path &base_output_directory = ".generated_libs" );

    GeneratedLibrarySet lib;
    T*           sym;
};

template<class T>
bool GeneratedSym<T>::need_init() const {
    return ! lib;
}

template<class T>
void GeneratedSym<T>::init( const std::string &name, const std::string &summary_of_parameters, const std::function<void( SrcWriter &sw )> &src_writer, const GeneratedLibrarySet::Path &base_output_directory ) {
    lib.init( name, summary_of_parameters, src_writer, base_output_directory );
    sym = reinterpret_cast<T *>( lib.symbol( name ) );
}
