#include "Node.h"

namespace parex {

Node::Node( Task *task ) : task( task ) {
    if ( task )
        ++task->cpt_use;
}

Node::~Node() {

}

} // namespace parex
