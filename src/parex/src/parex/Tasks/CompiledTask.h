#pragma once

#include "ComputableTask.h"
#include "../TypeFactory.h"
#include "../type_name.h"
#include "../TaskOut.h"
class SrcWriter;
class Src;

/**
*/
class CompiledTask : public ComputableTask {
public:
    /***/                  CompiledTask      ( std::vector<Rc<Task>> &&children, int priority = 0 );
    virtual               ~CompiledTask      ();

    virtual void           exec              () override;

    virtual void           get_src_content   ( Src &src, SrcWriter &sw ) = 0;
    virtual void           get_summary       ( std::ostream &os );
    virtual std::string    func_name         ();

    template<class F> void run_kernel_wrapper( const F &f );
    template<class F> void run_void_or_not   ( std::integral_constant<bool,0>, const F &func );
    template<class F> void run_void_or_not   ( std::integral_constant<bool,1>, const F &func );
    template<class T> void make_outputs      ( TaskOut<T> &&ret );
};

template<class T>
void CompiledTask::make_outputs( TaskOut<T> &&ret ) {
    TypeFactory &tf = type_factory_virtual();
    type = tf( type_name<T>() );
    data = ret.data;
}


template<class F>
void CompiledTask::run_void_or_not( std::integral_constant<bool,0>, const F &func ) {
    make_outputs( func( this ) );
}

template<class F>
void CompiledTask::run_void_or_not( std::integral_constant<bool,1>, const F &func ) {
    func( this );
}

template<class F>
void CompiledTask::run_kernel_wrapper( const F &func ) {
    constexpr bool void_ret = std::is_same<decltype( func( this ) ),void>::value;
    run_void_or_not( std::integral_constant<bool,void_ret>(), func );
}
