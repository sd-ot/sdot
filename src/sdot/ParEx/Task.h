#pragma once

#include "Kernel.h"
#include "Value.h"

namespace parex {

class Task {
public:
    /**/                Task                 ();

    void                get_front_rec        ( std::vector<Task *> &front );
    bool                children_are_computed() const;

    std::string         output_type;
    void*               output_data;
    bool                computed;

    Kernel              kernel;
    std::vector<Value>  inputs;

    static  std::size_t curr_op_id;
    mutable std::size_t cpt_use;
    mutable std::size_t op_id;
};

} // namespace parex
