#pragma once

#include "../data/TypeFactory.h"
#include "../data/TypeInfo.h"
#include "TaskOut.h"

namespace parex {

/**
*/
class ComputableTask : public Task {
public:
    /***/                  ComputableTask         ( std::vector<Rc<Task>> &&children, double priority = 0 );

    virtual bool           all_ch_computed        () const;
    virtual void           get_front_rec          ( std::map<int,std::vector<ComputableTask *>> &front ) override;
    virtual bool           is_computed            () const override;
    virtual void           prepare                (); ///< done before execution, each time there's something new in one child. Can be used to check the input types. By default: check that data are allocated in the same space.
    virtual void           exec                   () = 0;

    // helper to create the output from a class with a operator()( Task * ) method
    template<class F> void run_kernel_wrapper     ( const F &f );
    template<class F> void run_void_or_not        ( std::integral_constant<bool,0>, const F &func );
    template<class F> void run_void_or_not        ( std::integral_constant<bool,1>, const F &func );
    template<class T> void make_outputs           ( TaskOut<T> &&ret );

    //
    virtual void           check_input_same_memory(); ///<

    // graph data
    std::vector<Rc<Task>>  children;                  ///<
    double                 priority;                  ///<

    bool                   scheduled;                 ///<
    bool                   in_front;                  ///<
    bool                   computed;                  ///<
};

template<class T>
void ComputableTask::make_outputs( TaskOut<T> &&ret ) {
    output.type = type_factory_virtual( TypeInfo<T>::name() );
    output.data = ret.data;
    output.own = true;

    if ( ret.task )
        ret.task->output.own = false;
}


template<class F>
void ComputableTask::run_void_or_not( std::integral_constant<bool,0>, const F &func ) {
    make_outputs( func( this ) );
}

template<class F>
void ComputableTask::run_void_or_not( std::integral_constant<bool,1>, const F &func ) {
    func( this );
}

template<class F>
void ComputableTask::run_kernel_wrapper( const F &func ) {
    constexpr bool void_ret = std::is_same<decltype( func( this ) ),void>::value;
    run_void_or_not( std::integral_constant<bool,void_ret>(), func );
}

} // namespace parex
