#include "../plugins/GeneratedSymbolSet.h"
#include "../hardware/HwGraph.h"
#include "../plugins/Src.h"
#include "../utility/P.h"
#include "Type.h"

namespace parex {

Type::~Type() {
}

void Type::write_to_stream( std::ostream &os ) const {
    os << name;
}

void Type::destroy( void *data ) const {
    if ( ! destructor_func ) {
        static GeneratedSymbolSet destructors( ".generated_libs/destroy" );
        destructor_func = destructors.get_symbol<void(void *)>( [&]( SrcSet &sw ) {
            Src &src = sw.src( "destroy.cpp" );
            src.compilation_environment += compilation_environment;

            src << "extern \"C\" void destroy( void *data ) { delete reinterpret_cast<" << name << " *>( data ); }";
        }, /*summary*/ name, "destroy" );
    }

    destructor_func( data );
}

} // namespace parex
