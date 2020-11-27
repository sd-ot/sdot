#include "TypeFactoryRegistrar.h"

TypeFactoryRegistrar *last_type_factory_registrar = nullptr;

TypeFactoryRegistrar::TypeFactoryRegistrar( std::string name, std::vector<std::string> includes, std::vector<std::string> preliminaries ) : name( name ), includes( includes ), preliminaries( preliminaries ) {
    prev_type_factory_registrar = last_type_factory_registrar;
    last_type_factory_registrar = this;
}