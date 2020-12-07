#include "TypeFactoryRegister.h"
#include "TypeFactory.h"
#include "CppType.h"

#include "../utility/TODO.h"

namespace parex {

TypeFactory::TypeFactory() {
    // register the most common types. Less common one are handled by TypeFactoryRegistrar
    reg_type( "std::string", []( Type *type ) { type->compilation_environment.includes << "<string>"; } );

    // integers
    using A = std::array<std::string,2>;
    for( std::string s : { "8", "16", "32", "64" } ) {
        for( std::string p : { "S", "P" } ) {
            reg_type( p + "I" + s, [&]( Type *type ) {
                type->compilation_environment.includes << "<cstdint>";
                type->compilation_environment.preliminaries << "using " + p + "I" + s + " = std::" + ( p == "P" ? "u" : "" ) + "int" + s + "_t;";
            } );
        }
    }

    // floating point numbers
    for( A p : { A{ "FP32" , "float" }, A{ "FP64" , "double" } } ) {
        reg_type( p[ 0 ], [&]( Type *type ) {
            type->compilation_environment.preliminaries << "using " + p[ 0 ] + " = " + p[ 1 ] + ";";
        } );
    }
}

TypeFactory::~TypeFactory() {
}

Type *TypeFactory::operator()( const std::string &name ) {
    // types that have to be registered
    while( TypeFactoryRegister *r = last_type_factory_registrar ) {
        for( const std::string &name : r->names )
            type_map[ name ] = std::make_unique<CppType>( name, r->compilation_environment );
        last_type_factory_registrar = r->prev_type_factory_registrar;
    }

    // if not found, assumes it is a simple CppType without any include or preliminary
    auto iter = type_map.find( name );
    if ( iter == type_map.end() )
        iter = type_map.insert( iter, { name, make_type_info( name ) } );
    return iter->second.get();
}

Type *TypeFactory::reg_type( const std::string &name, const std::function<void(CppType &)> &f ) {
    auto iter = type_map.find( name );
    if ( iter == type_map.end() ) {
        auto res = std::make_unique<CppType>( name, CompilationEnvironment{} );
        f( *res );

        iter = type_map.insert( iter, { name, std::move( res ) } );
    }

    return iter->second.get();
}

std::unique_ptr<Type> TypeFactory::make_type_info( const std::string &name ) {
    // it's a pointer ?
    if ( name.size() && name[ name.size() - 1 ] == '*' )
        TODO; // return ;

    // it's a template ?
    auto s = name.find( '<' );
    if ( s != std::string::npos ) {
        Type *base_type = operator()( name.substr( 0, s ) );

        std::vector<Type *> sub_types;
        for( std::size_t b = s + 1, c = b, n = 0; c < name.size(); ++c ) {
            switch ( name[ c ] ) {
            case '>':
                if ( n-- == 0 ) {
                    sub_types.push_back( operator()( name.substr( b, c - b ) ) );
                    b = c + 1;
                }
                break;
            case ',':
                if ( n == 0 ) {
                    sub_types.push_back( operator()( name.substr( b, c - b ) ) );
                    b = c + 1;
                }
                break;
            case '<':
                ++n;
                break;
            default:
                break;
            }
        }

        return base_type->copy_with_sub_types( name, std::move( sub_types ) );
    }

    // else, we consider it as a simple cpp type
    return std::make_unique<CppType>( name, CompilationEnvironment{} );
}

} // namespace parex
