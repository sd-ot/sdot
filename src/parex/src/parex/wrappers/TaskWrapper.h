#pragma once

#include "../data/TypeInfo.h"
#include "../tasks/SrcTask.h"
#include "../utility/Rc.h"

namespace parex {

/***/
class TaskWrapper {
public:
    /**/         TaskWrapper    ( const Rc<Task> &task_ref );
    /**/         TaskWrapper    ( Rc<Task> &&task_ref );
    /**/         TaskWrapper    ( Task *task );
    /**/         TaskWrapper    ();

    void         write_to_stream( std::ostream &os ) const;
    Rc<Task>     to_string      ( double priority = 0 ) const;

    template     <class T>
    Rc<Task>     conv_to        () const { return conv_to( TypeInfo<T>::name() ); }
    Rc<Task>     conv_to        ( std::string type_name ) const;
    Rc<Task>     conv_to        ( Type *type ) const;

    Rc<Task>     task;
};

} // namespace parex
