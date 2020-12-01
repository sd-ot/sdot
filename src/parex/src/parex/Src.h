#pragma once

#include "CompilationEnvironment.h"
#include <filesystem>
#include <sstream>
#include <string>


/**
*/
class Src {
public:
    using                  Path                     = std::filesystem::path;
    using                  VUPath                   = VecUnique<Path>;

    /**/                   Src                      ( const CompilationEnvironment &compilation_environment );

    template<class T> Src& operator<<               ( const T &value ) { fout << value; return *this; }
    void                   write_to                 ( std::ostream &os ) const;

    CompilationEnvironment compilation_environment; ///<
    std::ostringstream     fout;                    ///<
};

