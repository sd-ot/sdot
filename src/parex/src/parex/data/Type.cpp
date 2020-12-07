#include "../plugins/GeneratedSymbolSet.h"
#include "../tasks/CompiledTask.h"
#include "../hardware/HwGraph.h"
#include "../utility/ERROR.h"
#include "../plugins/Src.h"
#include "../tasks/Task.h"
#include "../utility/P.h"
#include "Type.h"

namespace parex {

Type::Type( CompilationEnvironment &&compilation_environment, std::vector<Type *> &&sub_types ) :
    get_memories_func( nullptr ), destroy_func( nullptr ), compilation_environment( std::move( compilation_environment ) ), sub_types( std::move( sub_types ) ) {
}

Type::~Type() {
}

void Type::write_to_stream( std::ostream &os ) const {
    os << name;
}

Rc<Task> Type::conv_alloc_task( const Rc<Task> &task, Memory *mem ) const {
    struct ConvAlloc : CompiledTask {
        using CompiledTask::CompiledTask;

        virtual void get_src_content( Src &src, SrcSet &/*sw*/ ) override {
            src << "template<class Value,class Allocator>\n";
            src << "auto " << called_func_name() << "( parex::TaskOut<Value> &value, parex::TaskOut<Allocator> &allocator ) {\n";
            src << "    auto *res = parex::new_copy_in( *allocator, *value );\n";
            src << "    return parex::TaskOut<typename std::decay<decltype(*res)>::type>( res );\n";
            src << "}\n";
        }
    };

    Task *at = Task::new_src( Task::type_factory( mem->allocator_type() ), mem->allocator_data(), false );
    return new ConvAlloc( "ConvAlloc", { task, at } );
}

void Type::get_memories( VecUnique<Memory *> &memories, const void *data ) const {
    if ( ! get_memories_func ) {
        static GeneratedSymbolSet memories_symbol_set( ".generated_libs/destroy" );
        get_memories_func = memories_symbol_set.get_symbol<void(VecUnique<Memory *> &memories, const HwGraph *hw_graph, const void *)>( [&]( SrcSet &sw ) {
            Src &src = sw.src( "get_memories.cpp" );
            src.compilation_environment += compilation_environment;
            src.compilation_environment.includes << "<parex/utility/VecUnique.h>";
            src.compilation_environment.includes << "<parex/hardware/HwGraph.h>";

            gen_func_get_memories( src, sw );

            src << "extern \"C\" void get_memories_( parex::VecUnique<parex::Memory *> &memories, const parex::HwGraph *hw_graph, const void *data ) { get_memories( memories, hw_graph, reinterpret_cast<const " << name << " *>( data ) ); }";
        }, /*summary*/ name, "get_memories_" );
    }

    get_memories_func( memories, default_hw_graph(), data );
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
    // src.compilation_environment.includes << "<parex/hardware/default_CpuAllocator.h>";
    // src << "    memories << &default_CpuAllocator.memory;\n";
    src << "template<class T> void get_memories( parex::VecUnique<parex::Memory *> &memories, const parex::HwGraph *hw_graph, const T */*data*/ ) {\n";
    src << "}\n";
}

void Type::gen_func_destroy( Src &src, SrcSet &/*sw*/ ) const {
    src << "template<class T> void destroy( const T *data ) {\n";
    src << "    delete data;\n";
    src << "}\n";
}

} // namespace parex
