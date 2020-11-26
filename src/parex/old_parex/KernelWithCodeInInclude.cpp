#include "KernelWithCodeInInclude.h"
#include "SrcWriter.h"
#include <ostream>

KernelWithCodeInInclude::KernelWithCodeInInclude( const Path &path ) : path( path ) {
}

KernelWithCodeInInclude::Path KernelWithCodeInInclude::output_directory( const std::string &/*parameters*/ ) const {
    return "objects" / path;
}

void KernelWithCodeInInclude::write_to_stream( std::ostream &os ) const {
    os << path.string();
}

void KernelWithCodeInInclude::get_summary( std::ostream &os ) const {
    os << "KernelWithInclude\n" << path.string();
}

void KernelWithCodeInInclude::make_srcs( SrcWriter &ff ) const {
    ff.default_includes.push_back( "<" + path.string() + ">" );
    KernelWithCompiledCode::make_srcs( ff );
}

std::string KernelWithCodeInInclude::func_name( const std::string &/*parameters*/ ) const {
    return path.stem().string();
}
