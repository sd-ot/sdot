#pragma once

#include "TypeFactory.h"
#include "type_name.h"
#include "TaskOut.h"

/**
*/
class ComputableTask : public Task {
public:
    /***/                  ComputableTask    ( std::vector<Rc<Task>> &&children, double priority = 0 );

    virtual bool           all_ch_computed   () const;
    virtual void           get_front_rec     ( std::map<int,std::vector<ComputableTask *>> &front ) override;
    virtual bool           is_computed       () const override;
    virtual void           exec              () = 0;

    // helper to create the output from a class with a operator()( Task * ) method
    template<class F> void run_kernel_wrapper( const F &f );
    template<class F> void run_void_or_not   ( std::integral_constant<bool,0>, const F &func );
    template<class F> void run_void_or_not   ( std::integral_constant<bool,1>, const F &func );
    template<class T> void make_outputs      ( TaskOut<T> &&ret );

    // graph data
    std::vector<Rc<Task>>  children;         ///<
    double                 priority;         ///<

    bool                   scheduled;        ///<
    bool                   in_front;         ///<
    bool                   computed;         ///<
};

template<class T>
void ComputableTask::make_outputs( TaskOut<T> &&ret ) {
    output_type = type_factory_virtual( type_name<T>() );
    output_data = ret.data;
    output_own = true;

    if ( ret.task )
        ret.task->output_own = false;
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
