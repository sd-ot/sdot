#pragma once

#include "TypeInfo.h"
#include <memory>
#include <string>
#include <map>

/**
*/
class TypeInfoFactory {
public:
    /**/                      TypeInfoFactory();

    TypeInfo                 *operator()     ( const std::string &name );

private:
    using                     TypeMap        = std::map<std::string,std::unique_ptr<TypeInfo>>;

    std::unique_ptr<TypeInfo> make_type_info ( const std::string &name );

    TypeMap                   type_map;      ///<
};

extern TypeInfoFactory type_info_factory;
