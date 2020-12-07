#pragma once

#include "Type.h"
#include <memory>
#include <map>

namespace parex {

/**
*/
class TypeFactory {
public:
    using         TemplateFunc  = std::function<Type *( const std::string &name, const std::string &base_name, TypeFactory &tf, const std::vector<std::string> &parameters )>;

    /**/          TypeFactory   ();
    virtual      ~TypeFactory   ();

    virtual Type* operator()    ( const std::string &name );

    void          reg_template  ( const std::string &name, TemplateFunc &&f ); ///< f is stored to be called to construct the type if not already created

    Type*         reg_type      ( const std::string &name, const std::function<Type *( const std::string &name )> &f ); ///< f is called to construct the type if not already created
    Type*         reg_type      ( const std::string &name, const std::function<void( Type * )> &f = []( Type * ) {} ); ///< f is called to construct the type if not already created

private:
    using         TemplateMap   = std::map<std::string,TemplateFunc>;
    using         TypeMap       = std::map<std::string,std::unique_ptr<Type>>;

    Type*         make_type_info( const std::string &name );
    Type*         get_type_rec  ( const std::string &name );

    TemplateMap   template_map; ///<
    TypeMap       type_map;     ///<
};

} // namespace parex
