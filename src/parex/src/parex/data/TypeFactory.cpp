#include "TypeFactoryRegister.h"
#include "TypeFactory.h"

#include "../utility/TODO.h"
#include "../utility/P.h"

namespace parex {

TypeFactory::TypeFactory() {
    // register the most common types. Less common one are handled by TypeFactoryRegistrar
    reg_type( "std::string", []( Type *type ) { type->compilation_environment.includes << "<string>"; } );

    // integers
    for( std::string s : { "8", "16", "32", "64" } ) {
        for( std::string p : { "S", "P" } ) {
            reg_type( "parex::" + p + "I" + s, [&]( Type *type ) {
                type->compilation_environment.includes << "<cstdint>";
                type->compilation_environment.preliminaries << "namespace parex { using " + p + "I" + s + " = std::" + ( p == "P" ? "u" : "" ) + "int" + s + "_t; }";
            } );
        }
    }

    // floating point numbers
    using A = std::array<std::string,2>;
    for( A p : { A{ "parex::FP32" , "float" }, A{ "parex::FP64" , "double" } } ) {
        reg_type( p[ 0 ], [&]( Type *type ) {
            type->compilation_environment.preliminaries << "using " + p[ 0 ] + " = " + p[ 1 ] + ";";
        } );
    }
}

TypeFactory::~TypeFactory() {
}

Type *TypeFactory::reg_type( const std::string &name, const std::function<Type *( const std::string &name )> &f ) {
    // if not in instantiated types, create a new one using f, and complete attributes with base information
    auto iter = type_map.find( name );
    if ( iter == type_map.end() ) {
        Type *res = f( name );

        CompilationEnvironment ce;
        for( Type *sub_type : res->sub_types )
            ce += sub_type->compilation_environment;
        ce += res->compilation_environment;
        std::swap( res->compilation_environment, ce );

        if ( res->base_name.empty() )
            res->base_name = name;

        if ( res->name.empty() )
            res->name = name;

        iter = type_map.insert( iter, { name, std::unique_ptr<Type>{ res } } );
    }

    // ptr on instantiated type
    return iter->second.get();
}

Type *TypeFactory::reg_type( const std::string &name, const std::function<void( Type * )> &f ) {
    return reg_type( name, [&]( const std::string &/*name*/ ) {
        Type *res = new Type;
        f( res );

        return res;
    } );
}

void TypeFactory::reg_template( const std::string &name, TypeFactory::TemplateFunc &&f ) {
    template_map.insert( { name, std::move( f ) } );
}

Type *TypeFactory::operator()( const std::string &name ) {
    // types that have to be registered
    for( ; TypeFactoryRegister *r = last_type_factory_registrar; last_type_factory_registrar = r->prev_type_factory_registrar )
        r->reg( *this );

    // already in in instantiated types ?
    auto iter = type_map.find( name );
    if ( iter != type_map.end() )
        return iter->second.get();

    // -> create a new one
    return make_type_info( name );
}

Type *TypeFactory::make_type_info( const std::string &name ) {
    // it's a pointer ?
    if ( name.size() && name[ name.size() - 1 ] == '*' )
        TODO; // return ;

    // if not a template, we make a type with default args
    auto s = name.find( '<' );
    if ( s == std::string::npos )
        return reg_type( name );

    // template type => get parameters
    std::string base_name = name.substr( 0, s );
    std::vector<std::string> parameters;
    for( std::size_t b = s + 1, c = b, n = 0; c < name.size(); ++c ) {
        auto reg_parm = [&]() {
            parameters.push_back( name.substr( b, c - b ) );
            b = c + 1;
        };

        switch ( name[ c ] ) {
        case '>': reg_parm(); --n; break;
        case ',': reg_parm(); break;
        case '<': ++n; break;
        default : break;
        }
    }

    // if not template factory, create a base type (without subtypes)
    auto iter = template_map.find( base_name );
    if ( iter == template_map.end() ) {
        return reg_type( name, [&]( Type *res ) {
            res->parameters = parameters;
            res->base_name = base_name;
        } );
    }

    // else, created a type using the factory and register it
    return reg_type( name, [&]( const std::string &/*name*/ ) {
        Type *res = iter->second( name, base_name, *this, parameters );
        res->parameters = parameters;
        res->base_name = base_name;
        return res;
    } );
}

} // namespace parex
