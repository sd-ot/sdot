#pragma once

#include "CompiledTask.h"

namespace parex {

/***/
class GenericArithmeticOperation : public CompiledTask {
public:
    /**/         GenericArithmeticOperation( std::string name_op, std::vector<Rc<Task>> &&children );

    virtual void get_src_content           ( Src &src, SrcSet &sw ) override;

    std::string  name_op;
};

} // namespace parex
