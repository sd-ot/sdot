#pragma once

#include "../plugins/CompilationEnvironment.h"
#include "TypeFactory.h"
#include <variant>

namespace parex {

/**
*/
class TypeFactoryRegister {
public:
    using                    TNBFP               = std::function<Type*(const std::string &name, const std::string &base_name, TypeFactory &tf, const std::vector<std::string> &parameters)>;
    using                    TFP                 = std::function<Type*(TypeFactory &tf, const std::vector<std::string> &parameters)>;
    using                    FTN                 = std::function<Type*(const std::string &name)>;
    using                    FuncVariant         = std::variant<FTN,TFP,TNBFP>;

    /**/                     TypeFactoryRegister( std::vector<std::string> names, FuncVariant &&f );
    void                     reg                ( TypeFactory &tf );

    TypeFactoryRegister*     prev_type_factory_registrar;
    FuncVariant              func_variant;
    std::vector<std::string> names;
};

extern TypeFactoryRegister *last_type_factory_registrar;

} // namespace parex
