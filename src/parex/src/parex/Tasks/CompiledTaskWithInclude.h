#pragma once

#include "CompiledTask.h"

/**
*/
class CompiledTaskWithInclude : public CompiledTask {
public:
    /**/                CompiledTaskWithInclude( const Path &src_path, std::vector<Rc<Task>> &&children, int priority = 0 );

    virtual void        write_to_stream        ( std::ostream &os ) const override;
    virtual void        get_src_content        ( Src &src, SrcWriter &sw ) override;
    virtual std::string func_name              () override;

    Path                src_path;              ///<
};

