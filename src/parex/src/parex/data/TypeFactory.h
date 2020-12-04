#pragma once

#include "Type.h"
#include <map>

namespace parex {
class CppType;

/**
*/
class TypeFactory {
public:
    /**/          TypeFactory   ();
    virtual      ~TypeFactory   ();

    virtual Type* operator()    ( const std::string &name );

    Type*         reg_cpp_type  ( const std::string &name, const std::function<void(CppType &)> &f );

private:
    using         TypePtr       = std::unique_ptr<Type>;
    using         TypeMap       = std::map<std::string,TypePtr>;

    TypePtr       make_type_info( const std::string &name );
    Type*         get_type_rec  ( const std::string &name );

    TypeMap       type_map;     ///<
};

} // namespace parex
