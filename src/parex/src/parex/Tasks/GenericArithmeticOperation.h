#pragma once

#include "../CompiledTask.h"

/***/
class GenericArithmeticOperation : public CompiledTask {
public:
    /**/         GenericArithmeticOperation( std::string name_op, std::vector<Rc<Task>> &&children );

    virtual void write_to_stream           ( std::ostream &os ) const override;
    virtual void get_src_content           ( Src &src, SrcSet &sw ) override;

    std::string  name_op;
};

