#include "CompiledIncludeTask.h"
#include "TaskWrapper.h"
#include "Scheduler.h"
#include <limits>

namespace parex {

TaskWrapper::TaskWrapper( const Rc<Task> &task ) : task( task ) {
}

TaskWrapper::TaskWrapper( Rc<Task> &&task ) : task( std::move( task ) ) {
}

TaskWrapper::TaskWrapper( Task *task ) : task( task ) {
}

void TaskWrapper::write_to_stream( std::ostream &os ) const {
    Rc<Task> ts = to_string( std::numeric_limits<double>::max() );
    scheduler.append( ts );
    scheduler.run();

    os << *reinterpret_cast<const std::string *>( ts->output_data );
}

Rc<Task> TaskWrapper::to_string( double priority ) const {
    return new CompiledIncludeTask( "parex/kernels/to_string.h", { task }, {}, priority );
}

Rc<Task> TaskWrapper::conv_to( Type *type ) const {
    return static_cast<Task *>( new CompiledIncludeTask( "parex/kernels/conv_to.h", {
        static_cast<Task *>( new SrcTask( type, nullptr, false ) ),
        task
    }, {} ) );
}

Rc<Task> TaskWrapper::conv_to( std::string type_name ) const {
    return conv_to( Task::type_factory( type_name ) );
}


} // namespace parex
