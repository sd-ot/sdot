#pragma once

#include "Task.h"

namespace parex {

/***/
class Node {
public:
    /**/  Node();

    template<class ...Args>
    Node  New ( const Kernel &kernel, Args&& ...args );

    Task* task;
};

template<class ...Args>
Node Node::New( const Kernel &kernel, Args&& ...args ) {

}

} // namespace parex
