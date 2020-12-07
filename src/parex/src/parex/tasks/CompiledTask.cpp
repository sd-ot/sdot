#include "../plugins/GeneratedSymbolSet.h"
#include "CompiledTask.h"

namespace parex {

CompiledTask::CompiledTask( std::string &&name, std::vector<Rc<Task>> &&children, double priority ) : Task( std::move( name ), std::move( children ), priority ) {
}

void CompiledTask::exec() {
    // find or create in a static map
    static GeneratedSymbolSet gls;
    auto *func = gls.get_symbol<void( CompiledTask *)>( [&]( SrcSet &sw ) {
        Src &src = sw.src( called_func_name() + ".cpp" );

        src.compilation_environment.include_directories << PAREX_DIR "/src";
        src.compilation_environment.cpp_flags << "-std=c++17" << "-g";
        src.compilation_environment.includes << "<parex/tasks/CompiledTask.h>";

        // includes for types
        for( const Rc<Task> &ch : children )
            src.compilation_environment += ch->output.type->compilation_environment;

        // surdefined part
        get_src_content( src, sw );

        // generic part
        src << "\n";
        src << "namespace {\n";
        src << "    struct KernelWrapper {\n";
        src << "        auto operator()( parex::Task *task ) const {\n";
        for( std::size_t num_child = 0; num_child < children.size(); ++num_child )
            src << "            parex::TaskOut<" << children[ num_child ]->output.type->name << "> arg_" << num_child << "( task->move_child( " << num_child << " ) );\n";
        src << "            return " << called_func_name() << "(";
        for( std::size_t num_child = 0; num_child < children.size(); ++num_child )
            src << ( num_child ? ", " : " " ) << "arg_" << num_child;
        src << " );\n";
        src << "        }\n";
        src << "    };\n";
        src << "}\n";

        src << "\n";
        src << "extern \"C\" void " << exported_func_name() << "( parex::CompiledTask *task ) {\n";
        src << "    task->run_kernel_wrapper( KernelWrapper() );\n";
        src << "}\n";
    }, summary(), exported_func_name() );

    // call
    func( this );
}

std::string CompiledTask::exported_func_name() {
    return "kernel";
}

std::string CompiledTask::called_func_name() {
    return exported_func_name();
}

std::string CompiledTask::summary() {
    return {};
}

} // namespace parex
