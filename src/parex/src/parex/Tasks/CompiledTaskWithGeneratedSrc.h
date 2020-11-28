#pragma once

#include "CompiledTask.h"
#include "Src.h"

/**
*/
class CompiledTaskWithGeneratedSrc : public CompiledTask {
public:
    using        CodeGenFunc                 = std::function<void( Src &src, SrcWriter &sw )>;

    /***/        CompiledTaskWithGeneratedSrc( const std::string task_name, std::vector<Rc<Task>> &&children, CodeGenFunc &&code_gen_func, int priority = 0 );
    virtual void get_src_content             ( Src &src, SrcSet &sw ) override;
    virtual void write_to_stream             ( std::ostream &os ) const override;

    CodeGenFunc  code_gen_func;
    std::string  task_name;
};

