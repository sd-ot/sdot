#pragma once

#include "Kernel.h"
#include "Task.h"

namespace parex {

/***/
class Node {
public:
    template<class ...Args>
    /**/  Node( const Kernel &kernel, Args&& ...args );
    /**/  Node( Task *task = nullptr );
    /**/ ~Node();

    Task* task;
};

template<class ...Args>
Node::Node( const Kernel &kernel, Args&& .../*args*/ ) {

}

} // namespace parex
