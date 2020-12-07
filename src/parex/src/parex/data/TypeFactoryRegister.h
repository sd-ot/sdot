#pragma once

#include "../plugins/CompilationEnvironment.h"
#include "TypeFactory.h"
#include <variant>

namespace parex {

/**
*/
class TypeFactoryRegister {
public:
    // variants for template types
    using                    TNBFP               = std::function<Type*(const std::string &name, const std::string &base_name, TypeFactory &tf, const std::vector<std::string> &parameters)>;
    using                    TFP                 = std::function<Type*(TypeFactory &tf, const std::vector<std::string> &parameters)>;
    // variants for types
    using                    FTN                 = std::function<Type*(const std::string &name)>;
    using                    FVT                 = std::function<void(Type*)>;

    using                    FuncVariant         = std::variant<TNBFP,TFP,FTN,FVT>;

    /**/                     TypeFactoryRegister( std::vector<std::string> names, FuncVariant &&f );
    void                     reg                ( TypeFactory &tf );

    TypeFactoryRegister*     prev_type_factory_registrar;
    FuncVariant              func_variant;
    std::vector<std::string> names;
};

extern TypeFactoryRegister *last_type_factory_registrar;

} // namespace parex
