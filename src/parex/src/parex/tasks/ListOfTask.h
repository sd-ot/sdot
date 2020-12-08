#pragma once

#include "Task.h"

namespace parex {

/***/
class ListOfTask {
public:
    ListOfTask &operator<<( const Rc<Task> &task ) { tasks.push_back( task ); return *this; }
    std::vector<Rc<Task>> tasks;
};

} // namespace parex
