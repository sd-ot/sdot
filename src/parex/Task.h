#pragma once

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
    /**/                             Task                 () { computed = false; in_front = false; kernel = nullptr; cpt_use = 0; op_id = 0; }
    /**/                            ~Task                 ();


    template<class T> static Task*   owning               ( T *ptr ); ///< Wrap a known source value. Takes ownership of ptr
    static Task*                     call                 ( Kernel *kernel, std::vector<TaskRef> &&children = {} );

    bool                             children_are_computed() const;
    void                             get_front_rec        ( std::vector<Task *> &front );

    template<class F,class...A> void run                  ( const F &func, A& ...args );

    template<class F,class...A> void run_void_or_not      ( std::integral_constant<bool,0>, const F &func, A& ...args );
    template<class F,class...A> void run_void_or_not      ( std::integral_constant<bool,1>, const F &func, A& ...args );

    bool                             in_front;
    bool                             computed;
    std::vector<Output>              outputs;

    std::vector<TaskRef>             children;
    std::vector<Task *>              parents;
    Kernel*                          kernel;

    static  std::size_t              curr_op_id;
    mutable std::size_t              cpt_use;
    mutable std::size_t              op_id;
};

template<class T>
Task *Task::owning( T *ptr ) {
    Task *res = new Task;
    res->outputs.push_back( Output{ type_name( ptr ), ptr } );
    res->computed = true;
    return res;
}

template<class F,class...A>
void Task::run( const F &func, A& ...args ) {
    constexpr bool void_ret = std::is_same<decltype( func( args... ) ),void>::value;
    run_void_or_not( std::integral_constant<bool,void_ret>(), func, args... );
}

template<class F,class...A>
void Task::run_void_or_not( std::integral_constant<bool,0>, const F &func, A& ...args ) {
    auto ret = func( args... );
    outputs.push_back( { type_name( ret ), ret } );
}

template<class F,class...A>
void Task::run_void_or_not( std::integral_constant<bool,1>, const F &func, A& ...args ) {
    func( args... );
}

} // namespace parex
