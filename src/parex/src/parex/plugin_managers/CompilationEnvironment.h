#pragma once

#include "../utility/VecUnique.h"
#include <string>

/*
*/
class CompilationEnvironment {
public:
    using VS        = VecUnique<std::string>;

    void  operator+=( const CompilationEnvironment &that ) {
        cxx << that.cxx;
        includes << that.includes;
        cpp_flags << that.cpp_flags;
        libraries << that.libraries;
        preliminaries << that.preliminaries;
        cmake_packages << that.cmake_packages;
        include_directories << that.include_directories;
    }

    VS    cxx;
    VS    includes;
    VS    cpp_flags;
    VS    libraries;
    VS    preliminaries;
    VS    cmake_packages;
    VS    include_directories;
};
