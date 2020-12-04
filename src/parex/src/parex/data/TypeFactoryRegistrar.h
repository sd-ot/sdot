#pragma once

#include "../plugins/CompilationEnvironment.h"

namespace parex {

/**
*/
class TypeFactoryRegistrar {
public:
    /**/                     TypeFactoryRegistrar( std::vector<std::string> names, const CompilationEnvironment &compilation_environment );

    TypeFactoryRegistrar*    prev_type_factory_registrar;
    CompilationEnvironment   compilation_environment;
    std::vector<std::string> names;
};

extern TypeFactoryRegistrar *last_type_factory_registrar;

} // namespace parex
