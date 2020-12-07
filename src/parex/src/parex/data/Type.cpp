#include "../plugins/GeneratedSymbolSet.h"
#include "../hardware/HwGraph.h"
#include "../plugins/Src.h"
#include "../utility/P.h"
#include "Type.h"

namespace parex {

Type::Type( const std::string &name, CompilationEnvironment &&compilation_environment, const std::string &base_name, std::vector<Type *> &&sub_types ) :
    compilation_environment( std::move( compilation_environment ) ), destroy_func( nullptr ), base_name( base_name.empty() ? name : base_name ), sub_types( std::move( sub_types ) ), name( name ) {
}

Type::~Type() {
}

void Type::write_to_stream( std::ostream &os ) const {
    os << name;
}

void Type::memories( VecUnique<Type::Memory> &memories, const void *data ) const {
    if ( ! memories_func ) {
        static GeneratedSymbolSet memories_symbol_set( ".generated_libs/destroy" );
        memories_func = memories_symbol_set.get_symbol<void(VecUnique<Type::Memory> &memories, const HwGraph *hw_graph, const void *)>( [&]( SrcSet &sw ) {
            Src &src = sw.src( "memories.cpp" );
            src.compilation_environment += compilation_environment;
            src.compilation_environment.includes << "<parex/Data/Type.h>";

            gen_func_memories( src, sw );

            src << "extern \"C\" void memories_( parex::VecUnique<parex::Type::Memory> &res, const void *data ) { memories( res, reinterpret_cast<const " << name << " *>( data ) ); }";
        }, /*summary*/ name, "memories_" );
    }

    memories_func( memories, hw_graph(), data );
}

void Type::destroy( void *data ) const {
    if ( ! destroy_func ) {
        static GeneratedSymbolSet destructors( ".generated_libs/destroy" );
        destroy_func = destructors.get_symbol<void(void *)>( [&]( SrcSet &sw ) {
            Src &src = sw.src( "destroy.cpp" );
            src.compilation_environment += compilation_environment;

            gen_func_destroy( src, sw );

            src << "extern \"C\" void destroy_( void *data ) { destroy( reinterpret_cast<" << name << " *>( data ) ); }";
        }, /*summary*/ name, "destroy_" );
    }

    destroy_func( data );
}

void Type::gen_func_destroy( Src &src, SrcSet &/*sw*/ ) const {
    src << "template<class T> void memories( parex::VecUnique<parex::Type::Memory> &res, const HwGraph *hw_graph, const T */*data*/ ) {\n";
    src << "    res << hw_graph->local_memory();\n";
    src << "}\n";

}

} // namespace parex
