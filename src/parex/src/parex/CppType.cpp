#include "GeneratedSymbolSet.h"
#include "CppType.h"
#include "P.h"

CppType::CppType( std::string name, VecUnique<std::string> include_directories, VecUnique<std::string> includes, VecUnique<std::string> preliminaries, std::vector<Type *> &&sub_types ) :
    include_directories( include_directories ), destructor_func( nullptr ), preliminaries( preliminaries ), sub_types( std::move( sub_types ) ), includes( includes ), name( name ) {
}

Type::UPType CppType::copy_with_sub_type( std::string name, std::vector<Type *> &&sub_types ) const {
    return std::make_unique<CppType>( name, include_directories, includes, preliminaries, std::move( sub_types ) );
}

void CppType::for_each_include_directory( const std::function<void (std::string)> &cb ) const {
    for( Type *sub_type : sub_types )
        sub_type->for_each_include_directory( cb );
    for( const auto &p : include_directories )
        cb( p );
}

void CppType::for_each_prelim( const std::function<void(std::string)> &cb ) const {
    for( Type *sub_type : sub_types )
        sub_type->for_each_prelim( cb );
    for( const auto &p : preliminaries )
        cb( p );
}

void CppType::for_each_include( const std::function<void(std::string)> &cb ) const {
    for( Type *sub_type : sub_types )
        sub_type->for_each_include( cb );
    for( const auto &p : includes )
        cb( p );
}

void CppType::write_to_stream( std::ostream &os ) const {
    os << name;
}

std::string CppType::cpp_name() const {
    return name;
}

void CppType::destroy( void *data ) const {
    if ( ! destructor_func ) {
        static GeneratedSymbolSet destructors( ".generated_libs/destroy" );
        destructor_func = destructors.get_symbol<void(void *)>( [&]( SrcSet &sw ) {
            Src &src = sw.src( "destroy.cpp" );
            add_needs_in( src );

            src << "extern \"C\" void destroy( void *data ) { delete reinterpret_cast<" << cpp_name() << " *>( data ); }";
        }, /*summary*/ cpp_name(), "destroy" );
    }

    destructor_func( data );
}
