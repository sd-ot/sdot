#include "TypeFactoryRegister.h"

namespace parex {

TypeFactoryRegister *last_type_factory_registrar = nullptr;

static void reg_() {

}

TypeFactoryRegister::TypeFactoryRegister( std::vector<std::string> names, FuncVariant &&f ) : func_variant( std::move( tf ) ), names( names ) {
    prev_type_factory_registrar = last_type_factory_registrar;
    last_type_factory_registrar = this;
}

void TypeFactoryRegister::reg( TypeFactory &tf ) {
    //reg_();
}

} // namespace parex
