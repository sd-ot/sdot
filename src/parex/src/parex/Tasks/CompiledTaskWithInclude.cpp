#include "CompiledTaskWithInclude.h"
#include "../Src.h"

CompiledTaskWithInclude::CompiledTaskWithInclude( const Path &src_path, std::vector<Rc<Task>> &&children, int priority ) : CompiledTask( std::move( children ), priority ), src_path( src_path ) {
}

void CompiledTaskWithInclude::write_to_stream( std::ostream &os ) const {
    os << src_path.string();
}

std::string CompiledTaskWithInclude::func_name() {
    std::string name = src_path.string();
    name = name.substr( name.rfind( '/' ) + 1 );
    if ( name.find( '.' ) != std::string::npos )
        name = name.substr( 0, name.find( '.' ) );
    return name;
}

void CompiledTaskWithInclude::get_src_content( Src &src, SrcWriter &/*sw*/ ) {
    src.includes << "<" + src_path.string() + ">";
}
