#pragma once

#include "CompiledTask.h"
#include "SrcSet.h"

/**
*/
class CompiledLambdaTask : public CompiledTask {
public:
    using                  SrcWriter          = std::function<void( Src &src, SrcSet &sw, const std::vector<Rc<Task>> &children )>;

    /***/                  CompiledLambdaTask( SrcWriter &&src_writer, std::vector<Rc<Task>> &&children, const std::string &called_func_name, double priority = 0, const std::string &summary = {} );

    virtual std::string    called_func_name  () override;
    virtual void           write_to_stream   ( std::ostream &os ) const override;
    virtual void           get_src_content   ( Src &src, SrcSet &sw ) override;
    virtual std::string    summary           () override;

    std::string            called_func_name_;
    SrcWriter              src_writer_;
    std::string            summary_;
};

