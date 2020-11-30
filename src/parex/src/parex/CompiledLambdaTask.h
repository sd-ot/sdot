#pragma once

#include "CompiledTask.h"
#include "SrcSet.h"

/**
*/
class CompiledLambdaTask : public CompiledTask {
public:
    using               StreamWriter         = std::function<void( std::ostream &os, const std::vector<Rc<Task>> &children )>;
    using               SrcWriter            = std::function<void( Src &src, SrcSet &sw, const std::vector<Rc<Task>> &children )>;

    /***/               CompiledLambdaTask   ( StreamWriter &&get_summary, SrcWriter &&src_writer, std::vector<Rc<Task>> &&children, StreamWriter &&called_func_name_writer = {}, double priority = 0 );
    /***/               CompiledLambdaTask   ( SrcWriter &&src_writer, std::vector<Rc<Task>> &&children, StreamWriter &&called_func_name_writer = {}, double priority = 0 );

    virtual std::string called_func_name     () override;
    virtual void        write_to_stream      ( std::ostream &os ) const override;
    virtual void        get_src_content      ( Src &src, SrcSet &sw ) override;
    virtual std::string summary              () override;

    StreamWriter        called_func_name_writer_;
    StreamWriter        summary_writer_;
    SrcWriter           src_writer_;
};

