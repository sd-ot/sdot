#include "TypeInfo/LocalTypeInfo.h"
#include "TypeInfoFactory.h"

TypeInfoFactory type_info_factory;

TypeInfoFactory::TypeInfoFactory() {
    type_map[ "SI32" ] = std::unique_ptr<TypeInfo>{ new LocalTypeInfo( "SI32", { "<cstdint>" }, "using SI32 = std::int32_t;" ) };
}

TypeInfo *TypeInfoFactory::operator()( const std::string &name ) {
    auto iter = type_map.find( name );
    if ( iter == type_map.end() )
        iter = type_map.insert( iter, { name, make_type_info( name ) } );
    return iter->second.get();
}

std::unique_ptr<TypeInfo> TypeInfoFactory::make_type_info( const std::string &name ) {
    return std::make_unique<LocalTypeInfo>( name );
}
