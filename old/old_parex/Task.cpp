#include "KernelWithCodeInInclude.h"
#include "VariantWithPtr.h"
#include "Task.h"

RcPtr<Task> Task::call( const Path &kernel_name, std::vector<RcPtr<Task>> &&inputs ) {
    return call( new KernelWithCodeInInclude( kernel_name ), std::move( inputs ) );
}

RcPtr<Task> Task::call( Kernel *kernel, std::vector<RcPtr<Task>> &&inputs ) {
    Task *res = new Task;

    for( const RcPtr<Task> &ch : inputs )
        ch->parents.push_back( res );

    res->kernel = kernel;
    res->children = std::move( inputs );
    return res;
}

void Task::set_output( Type type, const void *data, bool owned ) {
    output = new Output;
    output->variants = { new VariantWithPtr( type, const_cast<void *>( data ), owned ) };
    computed = true;
}

Task::Task() {
    scheduled = 0;
    in_front  = 0;
    computed  = 0;
    priority  = 0;
}

