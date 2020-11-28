#include "CompiledTaskLib.h"
#include "../Type.h"
#include <sstream>

CompiledTaskLib::CompiledTaskLib( const Path &src_path, const std::vector<Type *> &children_types ) : GeneratedLibrarySet( "compiled_task_kernels", summary( src_path, children_types ) ) {
}

void CompiledTaskLib::make_srcs( SrcSet &sw ) {
    Src &src = sw.src( "compiled_task_kernel.cpp" );
    //    type->for_each_include( [&]( std::string p ) { src.includes << p; } );
    //    type->for_each_prelim( [&]( std::string p ) { src.prelims << p; } );
    //    src << "extern \"C\" void destroy( void *data ) { delete reinterpret_cast<" << type->cpp_name() << " *>( data ); }";
}

std::string CompiledTaskLib::summary( const Path &src_path, const std::vector<Type *> &children_types ) {
    std::ostringstream res;
    res << src_path.string() << "\n" << children_types.size();
    for( Type *type : children_types )
        type->write_to_stream( res << "\n" );
    return res.str();
}
