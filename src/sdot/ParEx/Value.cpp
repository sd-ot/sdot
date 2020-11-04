#include "Scheduler.h"
#include "Value.h"
#include "Task.h"

#include <utility>

namespace parex {

Value::Value( const Kernel &kernel, std::vector<Value> &&values ) {
    task = new Task;
    task->kernel = kernel;
    task->values = std::move( values );
    task->cpt_use = 1;
}

Value::Value( const Value &that ) : serialized_value( that.serialized_value ), task( that.task ) {
    inc_ref( task );
}

Value::Value( Value &&that ) : serialized_value( std::move( that.serialized_value ) ), task( std::exchange( that.task, nullptr ) ) {
}

Value::Value() : task( nullptr ) {
}

Value::~Value() {
    dec_ref( task );
}

Value &Value::operator=( const Value &that ) {
    inc_ref( that.task );
    dec_ref( task );

    serialized_value = that.serialized_value;
    task = that.task;
    return *this;
}

Value &Value::operator=( Value &&that ) {
    serialized_value = std::exchange( that.serialized_value, {} );
    task = std::exchange( that.task, nullptr );
    return *this;
}

void Value::write_to_stream( std::ostream &os ) const {
    scheduler << Value( Kernel{ "write_to_stream" }, &os );
    scheduler.run();
}

void Value::inc_ref( Task *task ) {
    if ( task )
        ++task->cpt_use;
}

void Value::dec_ref( Task *task ) {
    if ( task && ! --task->cpt_use )
        delete task;
}

} // namespace parex
