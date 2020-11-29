#pragma once

#include "CompiledTask.h"
#include "../VecUnique.h"

/**
*/
class CompiledTaskWithInclude : public CompiledTask {
public:
    using               VUS                    = VecUnique<std::string>;

    /**/                CompiledTaskWithInclude( const Path &src_path, std::vector<Rc<Task>> &&children, double priority = 0 );

    virtual void        write_to_stream        ( std::ostream &os ) const override;
    virtual void        get_src_content        ( Src &src, SrcSet &sw ) override;
    virtual std::string func_name              () override;

    VUS                 include_directories;   ///<
    Path                src_path;              ///<
};

#define ABS_SRC_PATH( NAME ) \
    ( std::filesystem::path( __FILE__ ).parent_path() / NAME )
