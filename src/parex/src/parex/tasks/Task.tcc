#include "../data/TypeInfo.h"
#include "TaskOut.h"
#include "Task.h"

namespace parex {

template<class T>
Task *Task::new_src_from_ptr( T *data, bool own ) {
    return Task::new_src( Task::type_factory( TypeInfo<T>::name() ), data, own );
}

template<class T>
void Task::make_outputs( TaskOut<T> &&ret ) {
    output.type = type_factory_virtual( TypeInfo<T>::name() );
    output.data = ret.data;
    output.own = true;

    if ( ret.task )
        ret.task->output.own = false;
}


template<class F>
void Task::run_void_or_not( std::integral_constant<bool,0>, const F &func ) {
    make_outputs( func( this ) );
}

template<class F>
void Task::run_void_or_not( std::integral_constant<bool,1>, const F &func ) {
    func( this );
}

template<class F>
void Task::run_kernel_wrapper( const F &func ) {
    constexpr bool void_ret = std::is_same<decltype( func( this ) ),void>::value;
    run_void_or_not( std::integral_constant<bool,void_ret>(), func );
}

} // namespace parex
