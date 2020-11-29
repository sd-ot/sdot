#pragma once

#include "DynamicLibrary.h"

/**
*/
template<class T>
struct DynamicSymbol {
    /**/            DynamicSymbol( DynamicLibrary *library, const std::string &name ) : library( library ) { symbol = library->symbol<T>( name ); }
    /**/            DynamicSymbol() { symbol = nullptr; }
    operator        bool         () { return symbol; }

    DynamicLibrary* library;
    T*              symbol;
};
