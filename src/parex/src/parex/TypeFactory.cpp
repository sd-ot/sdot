#include "TypeFactoryRegistrar.h"
#include "TypeFactory.h"
#include "CppType.h"
#include "TODO.h"

TypeFactory::TypeFactory() {
    // register the most common types. Less common one are handled by TypeFactoryRegistrar
    type_map[ "std::string" ] = std::make_unique<CppType>( "std::string", VecUnique<std::string>{}, VecUnique<std::string>{ "<string>"  }, VecUnique<std::string>{} );

    // integers
    using A = std::array<std::string,2>;
    for( A p : { A{ "SI8" , "int8_t"  }, A{ "PI8" , "uint8_t"  },
                 A{ "SI16", "int16_t" }, A{ "PI16", "uint16_t" },
                 A{ "SI32", "int32_t" }, A{ "PI32", "uint32_t" },
                 A{ "SI64", "int64_t" }, A{ "PI64", "uint64_t" } } )
        type_map[ p[ 0 ] ] = std::make_unique<CppType>( p[ 0 ], VecUnique<std::string>{}, VecUnique<std::string>{ "<cstdint>" }, VecUnique<std::string>{ "using " + p[ 0 ] + " = std::" + p[ 1 ] + ";" } );

    // floating point numbers
    for( A p : { A{ "FP32" , "float" }, A{ "FP64" , "double" } } )
        type_map[ p[ 0 ] ] = std::make_unique<CppType>( p[ 0 ], VecUnique<std::string>{}, VecUnique<std::string>{}, VecUnique<std::string>{ "using " + p[ 0 ] + " = " + p[ 1 ] + ";" } );
}

TypeFactory::~TypeFactory() {
}

Type *TypeFactory::operator()( const std::string &name ) {
    // types that have to be registered
    while( TypeFactoryRegistrar *r = last_type_factory_registrar ) {
        type_map[ r->name ] = std::make_unique<CppType>( r->name, r->include_directories, r->includes, r->preliminaries );
        last_type_factory_registrar = r->prev_type_factory_registrar;
    }

    // if not found, assumes it is a simple CppType without any include or preliminary
    auto iter = type_map.find( name );
    if ( iter == type_map.end() )
        iter = type_map.insert( iter, { name, make_type_info( name ) } );
    return iter->second.get();
}

Type *TypeFactory::reg_cpp_type( const std::string &name, const std::function<void(CppType &)> &f ) {
    auto iter = type_map.find( name );
    if ( iter == type_map.end() ) {
        auto res = std::make_unique<CppType>( name );
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

        return base_type->copy_with_sub_type( name, std::move( sub_types ) );
    }

    // else, we consider it as a simple cpp type
    return std::make_unique<CppType>( name );
}

