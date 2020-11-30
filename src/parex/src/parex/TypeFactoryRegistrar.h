#pragma once

#include "VecUnique.h"
#include <string>

/**
*/
class TypeFactoryRegistrar {
public:
    /**/                   TypeFactoryRegistrar( std::string name, VecUnique<std::string> includes = {}, VecUnique<std::string> preliminaries = {}, VecUnique<std::string> include_directories = {} );

    std::string            name;
    VecUnique<std::string> includes;
    VecUnique<std::string> preliminaries;
    VecUnique<std::string> include_directories;

    TypeFactoryRegistrar*  prev_type_factory_registrar;
};

extern TypeFactoryRegistrar *last_type_factory_registrar;
