#pragma once

#include "type_name.h"
#include <ostream>
#include <vector>

namespace parex {
class TaskRef;
class Kernel;

/**
*/
class Task {
public:
    /**/                 Task                 () { computed = false; in_front = false; kernel = nullptr; cpt_use = 0; op_id = 0; }
    /**/                ~Task                 ();

    template<class T>
    static Task*         owning               ( T *ptr ) { Task *res = new Task; res->output_type = type_name( ptr ); res->output_data = ptr; res->computed = true; return res; } ///< known value. Takes ownership of ptr

    void                 get_front_rec        ( std::vector<Task *> &front );
    bool                 children_are_computed() const;

    std::string          output_type;
    void*                output_data;
    bool                 in_front;
    bool                 computed;

    std::vector<TaskRef> children;
    std::vector<Task *>  parents;
    Kernel*              kernel;

    static  std::size_t  curr_op_id;
    mutable std::size_t  cpt_use;
    mutable std::size_t  op_id;
};

} // namespace parex
