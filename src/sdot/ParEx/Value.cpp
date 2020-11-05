#include "Scheduler.h"
#include "Value.h"
#include "Task.h"

#include <utility>

namespace parex {

namespace  {
    void inc_ref( Task *task ) {
        if ( task )
            ++task->cpt_use;
    }

    void dec_ref( Task *task ) {
        if ( task && ! --task->cpt_use )
            delete task;
    }
}

Value::Value( const Kernel &kernel, std::vector<Value> &&inputs ) {
    task = new Task;
    task->kernel = kernel;
    task->inputs = std::move( inputs );
}

Value::Value( const std::string &type, void *data ) {
    task = new Task;
    task->output_type = type;
    task->output_data = data;
    task->computed = true;
}

Value::Value( const Value &that ) : task( that.task ) {
    inc_ref( task );
}

Value::Value( Value &&that ) : task( std::exchange( that.task, nullptr ) ) {
}

Value::Value() : task( nullptr ) {
}

Value::~Value() {
    dec_ref( task );
}

Value &Value::operator=( const Value &that ) {
    inc_ref( that.task );
    dec_ref( task );

    task = that.task;
    return *this;
}

Value &Value::operator=( Value &&that ) {
    task = std::exchange( that.task, nullptr );
    return *this;
}

void Value::write_to_stream( std::ostream &os ) const {
    scheduler << Value( Kernel{ "write_to_stream" }, { os, *this } );
    scheduler.run();
}

} // namespace parex
