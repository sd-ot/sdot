#pragma once

#include "CompilationEnvironment.h"

/**
*/
class TypeFactoryRegistrar {
public:
    /**/                   TypeFactoryRegistrar( std::string name, const CompilationEnvironment &compilation_environment );

    TypeFactoryRegistrar*  prev_type_factory_registrar;
    CompilationEnvironment compilation_environment;
    std::string            name;
};

extern TypeFactoryRegistrar *last_type_factory_registrar;
