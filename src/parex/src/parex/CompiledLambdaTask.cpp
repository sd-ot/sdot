#include "CompiledLambdaTask.h"

CompiledLambdaTask::CompiledLambdaTask( SrcWriter &&src_writer, std::vector<Rc<Task> > &&children, const std::string &called_func_name, double priority, const std::string &summary ) : CompiledTask( std::move( children ), priority ),
        called_func_name_( called_func_name ),
        src_writer_( std::move( src_writer ) ),
        summary_( summary ) {
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
    return summary_;
}
