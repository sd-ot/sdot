#pragma once

#include "support/StaticRange.h"
#include "type_name.h"
#include "Output.h"
#include <ostream>
#include <vector>

namespace parex {
class TaskRef;
class Kernel;
class Value;

/**
*/
class Task {
public:
    /**/                           Task                 () { computed = false; in_front = false; kernel = nullptr; cpt_use = 0; op_id = 0; }
    /**/                          ~Task                 ();


    template<class T> static Task* owning               ( T *ptr ); ///< Wrap a known source value. Takes ownership of ptr
    static TaskRef                 call_r               ( Kernel *kernel, std::vector<TaskRef> &&inputs = {} ); ///< can be used if only 1 output. Return output of the task
    static Task*                   call                 ( Kernel *kernel, const std::vector<TaskRef *> &outputs = {}, std::vector<TaskRef> &&inputs = {} );

    bool                           children_are_computed() const;
    void                           get_front_rec        ( std::vector<Task *> &front );

    template<class F> void         run                  ( const F &func, void **data );

    template<class F> void         run_void_or_not      ( std::integral_constant<bool,0>, const F &func, void **data );
    template<class F> void         run_void_or_not      ( std::integral_constant<bool,1>, const F &func, void **data );

    template<class... A> void      make_outputs         ( std::tuple<A*...> &&t );
    template<class A> void         make_outputs         ( A *t );

    bool                           in_front;
    bool                           computed;
    std::vector<Output>            outputs;

    std::vector<TaskRef>           children;
    std::vector<Task *>            parents;
    Kernel*                        kernel;

    static  std::size_t            curr_op_id;
    mutable std::size_t            cpt_use;
    mutable std::size_t            op_id;
};

template<class T>
Task *Task::owning( T *ptr ) {
    Task *res = new Task;
    res->outputs.push_back( Output{ type_name( ptr ), ptr } );
    res->computed = true;
    return res;
}

template<class... A>
void Task::make_outputs( std::tuple<A*...> &&ret ) {
    constexpr std::size_t s = std::tuple_size<std::tuple<A*...>>::value;
    outputs.resize( s );

    StaticRange<s>::for_each( [&]( auto n ) {
        outputs[ n.value ] = { type_name( std::get<n.value>( ret ) ), std::get<n.value>( ret ) };
    } );
}

template<class A>
void Task::make_outputs( A *ret ) {
    outputs.push_back( { type_name( ret ), ret } );
}


template<class F>
void Task::run_void_or_not( std::integral_constant<bool,0>, const F &func, void **data ) {
    make_outputs( func( this, data ) );
}

template<class F>
void Task::run_void_or_not( std::integral_constant<bool,1>, const F &func, void **data ) {
    func( this, data );
}

template<class F>
void Task::run( const F &func, void **data ) {
    constexpr bool void_ret = std::is_same<decltype( func( this, data ) ),void>::value;
    run_void_or_not( std::integral_constant<bool,void_ret>(), func, data );
}

} // namespace parex
