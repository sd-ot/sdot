#include "TypeFactoryRegistrar.h"

TypeFactoryRegistrar *last_type_factory_registrar = nullptr;

TypeFactoryRegistrar::TypeFactoryRegistrar( std::string name, const CompilationEnvironment &compilation_environment ) : compilation_environment( compilation_environment ), name( name ) {
    prev_type_factory_registrar = last_type_factory_registrar;
    last_type_factory_registrar = this;
}
