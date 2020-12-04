#pragma once

#include "../plugins/SrcSet.h"
#include "CompiledTask.h"

namespace parex {

/**
*/
class CompiledIncludeTask : public CompiledTask {
public:
    using                  Path               = std::filesystem::path;

    /***/                  CompiledIncludeTask( const Path &include_path, std::vector<Rc<Task>> &&children, const std::string &called_func_name = {}, double priority = 0, const std::string &summary = {} );

    virtual std::string    called_func_name  () override;
    virtual void           write_to_stream   ( std::ostream &os ) const override;
    virtual void           get_src_content   ( Src &src, SrcSet &sw ) override;
    virtual std::string    summary           () override;

    std::string            called_func_name_;
    Path                   include_path_;
    std::string            summary_;
};

} // namespace parex
