#include "Types/CppType.h"
#include "TypeFactory.h"

TypeFactory::TypeFactory() {
    type_map[ "std::string" ] = std::make_unique<CppType>( "std::string", std::vector<std::string>{ "<string>"  }, std::vector<std::string>{} );

    using A = std::array<std::string,2>;
    for( A p : { A{ "SI32", "int32_t" }, A{ "PI32", "uint32_t" } } )
        type_map[ p[ 0 ] ] = std::make_unique<CppType>( p[ 0 ], std::vector<std::string>{ "<cstdint>" }, std::vector<std::string>{ "using " + p[ 0 ] + " = std::" + p[ 1 ] + ";" } );
}

TypeFactory::~TypeFactory() {
}

Type *TypeFactory::operator()( const std::string &name ) {
    auto iter = type_map.find( name );
    if ( iter == type_map.end() )
        iter = type_map.insert( iter, { name, make_type_info( name ) } );
    return iter->second.get();
}

std::unique_ptr<Type> TypeFactory::make_type_info( const std::string &name ) {
    return std::make_unique<CppType>( name );
}

