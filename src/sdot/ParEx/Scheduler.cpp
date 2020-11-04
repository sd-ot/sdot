#include "KernelCode.h"
#include "Scheduler.h"
#include "Task.h"

namespace parex {

Scheduler scheduler;

Scheduler::Scheduler() {
}

Scheduler &Scheduler::operator<<( const Value &value ) {
    targets.push_back( value );
    return *this;
}

void Scheduler::run() {
    for( const Value &value : targets ) {
        if ( Task *task = value.get_task() ) {
            auto func = kernel_code( task->kernel );
            func();
        }
    }
}

} // namespace parex
