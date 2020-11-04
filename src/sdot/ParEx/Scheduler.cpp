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
    //
    std::vector<std::string> input_type;
    std::vector<void *> input_data;
    for( const Value &value : targets ) {
        if ( value.task ) {
            if ( std::size_t ni = value.task->inputs.size() ) {
                input_type.resize( ni );
                input_data.resize( ni );
                for( std::size_t i = 0; i < ni; ++i ) {
                    input_type[ i ] = value.task->inputs[ i ].task->output_type;
                    input_data[ i ] = value.task->inputs[ i ].task->output_data;
                }
            }

            auto func = kernel_code.func( value.task->kernel, input_type );
            func( input_data.data() );
        }
    }
}

} // namespace parex
