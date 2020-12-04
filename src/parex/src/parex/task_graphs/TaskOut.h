#ifndef PAREX_TaskOut__H
#define PAREX_TaskOut__H

#include "../utility/Rc.h"
#include "Task.h"

/**
  Reference to an output of a task
*/
template<class T>
class TaskOut {
public:
    /**/     TaskOut   ( Rc<Task> &&task, T *data );
    /**/     TaskOut   ( Rc<Task> &&task );
    /**/     TaskOut   ( TaskOut &&out );
    /**/     TaskOut   ( T *data = nullptr );

    T*       operator->() const;
    T&       operator* () const;

    Rc<Task> task;
    T*       data;
};


#include "TaskOut.tcc"

#endif // PAREX_TaskOut__H

