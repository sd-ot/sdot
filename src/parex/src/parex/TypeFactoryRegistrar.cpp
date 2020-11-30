#include "TypeFactoryRegistrar.h"

TypeFactoryRegistrar *last_type_factory_registrar = nullptr;

TypeFactoryRegistrar::TypeFactoryRegistrar( std::string name, VecUnique<std::string> includes, VecUnique<std::string> preliminaries, VecUnique<std::string> include_directories ) : name( name ), includes( includes ), preliminaries( preliminaries ), include_directories( include_directories ) {
    prev_type_factory_registrar = last_type_factory_registrar;
    last_type_factory_registrar = this;
}
