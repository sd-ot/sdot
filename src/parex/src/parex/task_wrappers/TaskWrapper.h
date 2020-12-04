#pragma once

#include "../output_types/type_name.h"
#include "../task_graphs/SrcTask.h"
#include "../utility/Rc.h"

namespace parex {

/***/
class TaskWrapper {
public:
    /**/         TaskWrapper    ( const Rc<Task> &task_ref );
    /**/         TaskWrapper    ( Rc<Task> &&task_ref );
    /**/         TaskWrapper    ( Task *task );
    /**/         TaskWrapper    ();

    TaskWrapper& operator=      ( const TaskWrapper &that ) = delete;
    TaskWrapper& operator=      ( TaskWrapper &&that ) = default;

    void         write_to_stream( std::ostream &os ) const;
    Rc<Task>     to_string      ( double priority = 0 ) const;

    template     <class T>
    Rc<Task>     conv_to        () const { return conv_to( type_name( S<T>() ) ); }
    Rc<Task>     conv_to        ( std::string type_name ) const;
    Rc<Task>     conv_to        ( Type *type ) const;

    Rc<Task> task;
};

} // namespace parex
