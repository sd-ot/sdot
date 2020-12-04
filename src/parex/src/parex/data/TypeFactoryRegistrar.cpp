#include "TypeFactoryRegistrar.h"

namespace parex {

TypeFactoryRegistrar *last_type_factory_registrar = nullptr;

TypeFactoryRegistrar::TypeFactoryRegistrar( std::vector<std::string> names, const CompilationEnvironment &compilation_environment ) : compilation_environment( compilation_environment ), names( names ) {
    prev_type_factory_registrar = last_type_factory_registrar;
    last_type_factory_registrar = this;
}

} // namespace parex
