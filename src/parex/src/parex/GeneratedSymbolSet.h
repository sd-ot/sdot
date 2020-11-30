#pragma once

#include "GeneratedLibrarySet.h"

/**

*/
class GeneratedSymbolSet {
public:
    template<class T>
    T* get_symbol( const std::function<void(SrcSet&)> &src_writer, const std::string &summary_of_parameters = {}, const std::string &symbol_name = "exported" );

private:
    GeneratedLibrarySet lib_set;
};
