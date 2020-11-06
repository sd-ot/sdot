#include "Scheduler.h"
#include "Kernel.h"
#include "Value.h"

namespace parex {

Value::Value( Kernel *kernel, const std::vector<Value> &children ) : Value( new Task ) {
    ref.task->kernel = kernel;

    ref.task->children.resize( children.size() );
    for( std::size_t i = 0; i < children.size(); ++i ) {
        children[ i ].ref.task->parents.push_back( ref.task );
        ref.task->children[ i ] = children[ i ].ref;
    }
}

void Value::write_to_stream( std::ostream &os ) const {
    scheduler << Value( new Kernel{ "write_to_stream" }, { Task::owning( &os ), *this } );
    scheduler.run();
}

} // namespace parex
