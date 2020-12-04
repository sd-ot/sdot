#include "CompiledIncludeTask.h"
#include "../utility/P.h"

namespace parex {

CompiledIncludeTask::CompiledIncludeTask( const Path &include_path, std::vector<Rc<Task> > &&children, const std::string &called_func_name, double priority, const std::string &summary ) : CompiledTask( std::move( children ), priority ),
        called_func_name_( called_func_name.empty() ? include_path.stem().string() : called_func_name ),
        include_path_( include_path ),
        summary_( summary ) {
}

std::string CompiledIncludeTask::called_func_name() {
    return called_func_name_;
}

void CompiledIncludeTask::write_to_stream( std::ostream &os ) const {
    os << called_func_name_;
}

void CompiledIncludeTask::get_src_content( Src &src, SrcSet & ) {
    src << "#include <" << include_path_.string() << ">";
}

std::string CompiledIncludeTask::summary() {
    return summary_;
}

} // namespace parex
