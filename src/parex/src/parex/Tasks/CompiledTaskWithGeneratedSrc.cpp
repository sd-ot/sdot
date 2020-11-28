#include "CompiledTaskWithGeneratedSrc.h"

CompiledTaskWithGeneratedSrc::CompiledTaskWithGeneratedSrc( const std::string task_name, std::vector<Rc<Task>> &&children, CodeGenFunc &&code_gen_func, int priority ) :
    CompiledTask( std::move( children ), priority ), code_gen_func( std::move( code_gen_func ) ), task_name( task_name ) {
}

void CompiledTaskWithGeneratedSrc::get_src_content( Src &src, SrcSet &sw ) {
    code_gen_func( src, sw );
}

void CompiledTaskWithGeneratedSrc::write_to_stream( std::ostream &os ) const {
    os << task_name;
}

