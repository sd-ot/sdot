#include "TypeFactoryRegistrar.h"
#include "Types/CppType.h"
#include "TypeFactory.h"

TypeFactory::TypeFactory() {
    // register the most common types. Less common one are handled by TypeFactoryRegistrar
    type_map[ "std::string" ] = std::make_unique<CppType>( "std::string", std::vector<std::string>{ "<string>"  }, std::vector<std::string>{} );

    using A = std::array<std::string,2>;
    for( A p : { A{ "SI32", "int32_t" }, A{ "PI32", "uint32_t" } } )
        type_map[ p[ 0 ] ] = std::make_unique<CppType>( p[ 0 ], std::vector<std::string>{ "<cstdint>" }, std::vector<std::string>{ "using " + p[ 0 ] + " = std::" + p[ 1 ] + ";" } );
}

TypeFactory::~TypeFactory() {
}

Type *TypeFactory::operator()( const std::string &name ) {
    for( ; last_type_factory_registrar; last_type_factory_registrar = last_type_factory_registrar->prev_type_factory_registrar ) {
        type_map[ last_type_factory_registrar->name ] = std::make_unique<CppType>(
            last_type_factory_registrar->name,
            last_type_factory_registrar->includes,
            last_type_factory_registrar->preliminaries
        );
    }

    // if not found, assumes it is a simple CppType without any include or preliminary
    auto iter = type_map.find( name );
    if ( iter == type_map.end() )
        iter = type_map.insert( iter, { name, make_type_info( name ) } );
    return iter->second.get();
}

std::unique_ptr<Type> TypeFactory::make_type_info( const std::string &name ) {
//    if ( name.find( '<' ) != std::string::npos ) {
//        return get_type_rec( name );
//    }
    return std::make_unique<CppType>( name );
}

