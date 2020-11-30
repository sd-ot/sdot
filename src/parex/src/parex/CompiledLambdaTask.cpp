#include "CompiledLambdaTask.h"

CompiledLambdaTask::CompiledLambdaTask( StreamWriter &&get_summary, SrcWriter &&src_writer, std::vector<Rc<Task> > &&children, StreamWriter &&called_func_name_writer, double priority ) : CompiledTask( std::move( children ), priority ),
        called_func_name_( called_func_name ),
        summary_writer_( std::move( get_summary ) ),
        src_writer_( std::move( src_writer ) ) {
}

CompiledLambdaTask::CompiledLambdaTask( SrcWriter &&src_writer, std::vector<Rc<Task>> &&children, StreamWriter &&called_func_name_writer, double priority ) :
    CompiledLambdaTask( []( std::ostream &, const std::vector<Rc<Task>> & ) {}, std::move( src_writer ), std::move( children ), called_func_name, priority ) {
}

std::string CompiledLambdaTask::called_func_name() {
    return called_func_name_.empty() ? CompiledTask::called_func_name() : called_func_name_;
}

void CompiledLambdaTask::write_to_stream( std::ostream &os ) const {
    os << called_func_name_;
}

void CompiledLambdaTask::get_src_content( Src &src, SrcSet &sw ) {
    src_writer_( src, sw, children );
}

std::string CompiledLambdaTask::summary() {
    std::ostringstream os;
    summary_writer_( os, children );
    return os.str();
}
