#include "GeneratedSymbolSet.h"
#include "CppType.h"
#include "P.h"

CppType::CppType( std::string name, const CompilationEnvironment &compilation_environment, std::vector<Type *> &&sub_types ) : destructor_func( nullptr ), sub_types( std::move( sub_types ) ), name( name ) {
    this->compilation_environment = compilation_environment;
}

Type::UPType CppType::copy_with_sub_type( std::string name, std::vector<Type *> &&sub_types ) const {
    return std::make_unique<CppType>( name, compilation_environment, std::move( sub_types ) );
}

void CppType::for_each_type_rec( const std::function<void(const Type *)> &cb ) const {
    for( Type *sub_type : sub_types )
        cb( sub_type );
    cb( this );
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
