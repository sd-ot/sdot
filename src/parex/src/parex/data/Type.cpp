#include "../plugins/GeneratedSymbolSet.h"
#include "../hardware/HwGraph.h"
#include "../plugins/Src.h"
#include "../utility/P.h"
#include "Type.h"

namespace parex {

Type::Type( CompilationEnvironment &&compilation_environment, std::vector<Type *> &&sub_types ) :
    compilation_environment( std::move( compilation_environment ) ), get_memories_func( nullptr ), destroy_func( nullptr ), sub_types( std::move( sub_types ) ) {
}

Type::~Type() {
}

void Type::write_to_stream( std::ostream &os ) const {
    os << name;
}

void Type::get_memories( VecUnique<hardware_information::Memory *> &memories, const void *data ) const {
    if ( ! get_memories_func ) {
        static GeneratedSymbolSet memories_symbol_set( ".generated_libs/destroy" );
        get_memories_func = memories_symbol_set.get_symbol<void(VecUnique<hardware_information::Memory *> &memories, const HwGraph *hw_graph, const void *)>( [&]( SrcSet &sw ) {
            Src &src = sw.src( "get_memories.cpp" );
            src.compilation_environment += compilation_environment;
            src.compilation_environment.includes << "<parex/utility/VecUnique.h>";
            src.compilation_environment.includes << "<parex/hardware/HwGraph.h>";

            gen_func_get_memories( src, sw );

            src << "extern \"C\" void get_memories_( parex::VecUnique<parex::hardware_information::Memory *> &memories, const parex::HwGraph *hw_graph, const void *data ) { get_memories( memories, hw_graph, reinterpret_cast<const " << name << " *>( data ) ); }";
        }, /*summary*/ name, "get_memories_" );
    }

    get_memories_func( memories, hw_graph(), data );
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

void Type::gen_func_get_memories( Src &src, SrcSet &/*sw*/ ) const {
    src << "template<class T> void get_memories( parex::VecUnique<parex::hardware_information::Memory *> &memories, const parex::HwGraph *hw_graph, const T */*data*/ ) {\n";
    src << "    memories << hw_graph->local_memory();\n";
    src << "}\n";
}

void Type::gen_func_destroy( Src &src, SrcSet &/*sw*/ ) const {
    src << "template<class T> void destroy( const T *data ) {\n";
    src << "    delete data;\n";
    src << "}\n";
}

} // namespace parex
