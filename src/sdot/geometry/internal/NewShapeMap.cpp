#include "NewShapeMap.h"

void NewShapeMap::exec() {

}

//namespace {
//std::vector<ElementaryPolytopInfo> elementary_polytop_info_list( std::istringstream shape_types ) {
//    std::vector<ElementaryPolytopInfo> res;
//    for( std::string name; shape_types >> name; )
//        res.push_back( name );
//    return res;
//}
//}


//// find or create in a static map
//static GeneratedLibrarySet gls;
//DynamicLibrary *lib = gls.get_library( [&]( SrcSet &sw ) {
//    Src &src = sw.src( called_func_name() + ".cpp" );
//    src.include_directories << PAREX_DIR "/src";
//    src.cpp_flags << "-std=c++17" << "-g3";
//    src.includes << "<parex/CompiledTask.h>";

//    // includes for types
//    for( const Rc<Task> &ch : children )
//        ch->output_type->add_needs_in( src );

//    // surdefined part
//    get_src_content( src, sw );

//    // generic part
//    src << "\n";
//    src << "namespace {\n";
//    src << "    struct KernelWrapper {\n";
//    src << "        auto operator()( ComputableTask *task ) const {\n";
//    for( std::size_t num_child = 0; num_child < children.size(); ++num_child )
//        src << "            TaskOut<" << children[ num_child ]->output_type->cpp_name() << "> arg_" << num_child << "( std::move( task->children[ " << num_child << " ] ) );\n";
//    src << "            return " << called_func_name() << "(";
//    for( std::size_t num_child = 0; num_child < children.size(); ++num_child )
//        src << ( num_child ? ", " : " " ) << "arg_" << num_child;
//    src << " );\n";
//    src << "        }\n";
//    src << "    };\n";
//    src << "}\n";

//    src << "\n";
//    src << "extern \"C\" void " << exported_func_name() << "( CompiledTask *task ) {\n";
//    src << "    task->run_kernel_wrapper( KernelWrapper() );\n";
//    src << "}\n";
//}, summary() );

//// execute the generated function
//auto *func = lib->symbol<void( CompiledTask *)>( exported_func_name() );
//func( this );
