#pragma once

#include "support/StaticRange.h"
#include "support/ASSERT.h"

#include "type_name.h"
#include "Output.h"
#include "Kernel.h"

#include <functional>
#include <ostream>
#include <vector>
#include <set>

namespace parex {
class TaskRef;
class Value;

/**
*/
class Task {
public:
    /**/                           Task                 () { is_target_in_scheduler = false; computed = false; in_front = false; ref_count = 0; op_id = 0; }
    /**/                          ~Task                 ();

    void                           write_to_stream      ( std::ostream &os ) const;
    bool                           move_arg             ( std::size_t num_arg, std::size_t num_out = 0 );
    bool                           move_arg             ( const std::vector<std::size_t> &num_arg );
    bool                           move_arg             ( const std::vector<std::size_t> &num_arg, const std::vector<std::size_t> &num_out );

    static Task*                   ref_type             ( const std::string type ); ///< make a S<Type>() object
    static Task*                   ref_num              ( int value ); ///< make a N<value>() object
    template<class T> static Task* ref_on               ( T *ptr, bool own = true ); ///< Wrap a known source value. Takes ownership of ptr
    static TaskRef                 call_r               ( const Kernel &kernel, std::vector<TaskRef> &&inputs = {} ); ///< can be used if only 1 output. Return output of the task
    static Task*                   call                 ( const Kernel &kernel, const std::vector<TaskRef *> &outputs = {}, std::vector<TaskRef> &&inputs = {} );

    static void                    display_graphviz     ( const std::vector<Task *> &tasks, std::string f = ".tasks.dot", const char *prg = nullptr );
    void                           for_each_rec         ( const std::function<void( Task * )> &f, std::set<Task *> &seen );

    bool                           children_are_computed() const;
    void                           get_front_rec        ( std::vector<TaskRef> &front );

    template<class F> void         run                  ( const F &func, void **data );

    template<class F> void         run_void_or_not      ( std::integral_constant<bool,0>, const F &func, void **data );
    template<class F> void         run_void_or_not      ( std::integral_constant<bool,1>, const F &func, void **data );

    template<class... A> void      make_outputs         ( std::tuple<A*...> &&t );
    template<class A> void         make_outputs         ( A *t );

    std::vector<Output>            outputs;

    std::vector<TaskRef>           children;
    std::vector<Task *>            parents;
    Kernel                         kernel;

    bool                           is_target_in_scheduler;
    static  std::size_t            curr_op_id;
    mutable std::size_t            ref_count;
    bool                           in_front;
    bool                           computed;
    mutable std::size_t            op_id;
};

template<class T>
Task *Task::ref_on( T *ptr, bool own ) {
    Task *res = new Task;
    res->outputs.emplace_back( type_name( ptr ), ptr, own );
    res->computed = true;
    return res;
}

template<class... A>
void Task::make_outputs( std::tuple<A*...> &&ret ) {
    constexpr std::size_t s = std::tuple_size<std::tuple<A*...>>::value;
    outputs.resize( s );

    StaticRange<s>::for_each( [&]( auto n ) {
        if ( outputs[ n.value ].type.empty() )
            outputs[ n.value ] = { type_name( std::get<n.value>( ret ) ), std::get<n.value>( ret ) };
        else
            ASSERT( std::get<n.value>( ret ) == nullptr, "" );
    } );
}

template<class A>
void Task::make_outputs( A *ret ) {
    outputs.resize( 1 );
    if ( outputs[ 0 ].type.empty() )
        outputs[ 0 ] = { type_name( ret ), ret };
    else
        ASSERT( ret == nullptr, "" );
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
