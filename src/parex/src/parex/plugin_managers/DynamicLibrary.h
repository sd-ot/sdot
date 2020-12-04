#pragma once

#include <dynalo/dynalo.hpp>
#include <filesystem>

/**
*/
class DynamicLibrary {
public:
    using                Path          = std::filesystem::path;

    /**/                 DynamicLibrary( const Path &path );
    template<class T> T *symbol        ( const std::string &symbol_name ) { return lib.get_function<T>( symbol_name ); }

private:
    dynalo::library      lib;
};

