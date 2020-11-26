#include "../GeneratedSym.h"
#include "CompiledTask.h"

CompiledTask::CompiledTask( std::vector<Rc<Task>> &&children, int priority ) : ComputableTask( std::move( children ) ) {
    this->priority = priority;
}

CompiledTask::~CompiledTask() {
}

void CompiledTask::exec() {
    std::vector<Type *> children_types( children.size() );
    for( std::size_t i = 0; i < children.size(); ++i )
        children_types[ i ] = children[ i ]->output_type();

    // summary
    std::ostringstream summary;
    get_summary( summary );
    summary << "\n" << children_types.size();
    for( Type *type : children_types )
        type->write_to_stream( summary << "\n" );

    // find or create in a static map
    static std::map<std::string,GeneratedSym<void(CompiledTask *)>> mctl;
    auto iter = mctl.find( summary.str() );
    if ( iter == mctl.end() ) {
        std::string func_name = this->func_name();

        iter = mctl.emplace_hint( iter, summary.str(), GeneratedSym<void(CompiledTask *)>{} );
        iter->second.init( func_name, summary.str(), [&]( SrcWriter &sw ) {
            Src &src = sw.src( func_name + ".cpp" );
            src.include_directories << PAREX_DIR "/src";
            src.cpp_flags << "-std=c++17" << "-g3";
            src.includes << "<parex/Tasks/CompiledTask.h>";

            // includes for types
            for( Type *type : children_types )
                type->for_each_include( [&]( std::string i ) { src.includes << i; } );
            for( Type *type : children_types )
                type->for_each_prelim( [&]( std::string i ) { src.prelims << i; } );

            // surdefined part
            get_src_content( src, sw );

            src << "\n";
            src << "namespace {\n";
            src << "    struct KernelWrapper {\n";
            src << "        auto operator()( CompiledTask *task ) const {\n";
            for( std::size_t num_child = 0; num_child < children_types.size(); ++num_child )
                src << "            TaskOut<" << children_types[ num_child ]->cpp_name() << "> arg_" << num_child << "( std::move( task->children[ " << num_child << " ] ) );\n";
            src << "            return " << func_name << "(";
            for( std::size_t num_child = 0; num_child < children_types.size(); ++num_child )
                src << ( num_child ? ", " : " " ) << "arg_" << num_child;
            src << " );\n";
            src << "        }\n";
            src << "    };\n";
            src << "}\n";

            src << "\n";
            src << "extern \"C\" void " << func_name << "( CompiledTask *task ) {\n";
            src << "    task->run_kernel_wrapper( KernelWrapper() );\n";
            src << "}\n";
        }, ".generated_libs/kernels" );
    }

    // execute the generated function
    iter->second.sym( this );
}

void CompiledTask::get_summary( std::ostream &os ) {
    write_to_stream( os );
}
