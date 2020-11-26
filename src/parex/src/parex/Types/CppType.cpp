#include "CppType.h"
#include "../P.h"

CppType::CppType( std::string name, std::vector<std::string> includes, std::vector<std::string> preliminaries ) : preliminaries( preliminaries ), includes( includes ), name( name ) {
}

void CppType::for_each_prelim( const std::function<void(std::string)> &cb ) const {
    for( const auto &p : preliminaries )
        cb( p );
}

void CppType::for_each_include( const std::function<void(std::string)> &cb ) const {
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
    if ( destructor_func.need_init() ) {
        destructor_func.init( "destroy", cpp_name(), [&]( SrcWriter &sw ) {
            Src &src = sw.src( "destroy.cpp" );
            for_each_include( [&]( std::string p ) { src.includes << p; } );
            for_each_prelim( [&]( std::string p ) { src.prelims << p; } );
            src << "extern \"C\" void destroy( void *data ) { delete reinterpret_cast<" << cpp_name() << " *>( data ); }";
        } );
    }
    destructor_func.sym( data );
}
