#pragma once

#include "Type.h"
#include <memory>
#include <string>
#include <map>

/**
*/
class TypeFactory {
public:
    /**/          TypeFactory   ();
    virtual      ~TypeFactory   ();

    virtual Type* operator()    ( const std::string &name );

private:
    using         TypePtr       = std::unique_ptr<Type>;
    using         TypeMap       = std::map<std::string,TypePtr>;

    TypePtr       make_type_info( const std::string &name );

    TypeMap       type_map;     ///<
};

