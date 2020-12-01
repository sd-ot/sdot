#pragma once

#include "VecUnique.h"
#include <string>

/*
*/
struct CompilationEnvironment {
    VecUnique<std::string> includes;
    VecUnique<std::string> preliminaries;
    VecUnique<std::string> include_directories;

    VecUnique<std::string> cmake_packages;
    VecUnique<std::string> cmake_libraries;
};
