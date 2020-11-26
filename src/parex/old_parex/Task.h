#pragma once

#include "type_name.h"
#include "Kernel.h"
#include "Output.h"
#include "RcPtr.h"
#include "Type.h"

#include <filesystem>
#include <vector>

/**
*/
class Task {
public:
    using                          Path              = std::filesystem::path;

    template<class T> static Task *from_data_ptr     ( const T *ptr, bool owned = true ) { Task *res = new Task; res->set_output( Type{ type_name<T>() }, ptr, owned ); return res; }
    static RcPtr<Task>             call              ( const Path &kernel_name, std::vector<RcPtr<Task>> &&inputs );
    static RcPtr<Task>             call              ( Kernel *kernel, std::vector<RcPtr<Task>> &&inputs );

    virtual void                   set_output        ( Type type, const void *data, bool owned = true );

    template<class F> void         run_kernel_wrapper( const F &func );
    template<class F> void         run_void_or_not   ( std::integral_constant<bool,0>, const F &func );
    template<class F> void         run_void_or_not   ( std::integral_constant<bool,1>, const F &func );
    template<class T> void         make_outputs      ( TypedVariant<T> *ret );


    RefCount                       ref_count;        ///<
    std::vector<RcPtr<Task>>       children;         ///<
    std::vector<Task *>            parents;          ///<
    Kernel*                        kernel;           ///<
    RcPtr<Output>                  output;           ///<

    bool                           scheduled;        ///<
    bool                           in_front;         ///<
    bool                           computed;         ///<
    int                            priority;         ///<

private:
    /**/                           Task              ();
};

//template<class... A>
//void Task::make_outputs( std::tuple<A*...> &&ret ) {
//    constexpr std::size_t s = std::tuple_size<std::tuple<A*...>>::value;
//    outputs.resize( s );

//    StaticRange<s>::for_each( [&]( auto n ) {
//        if ( outputs[ n.value ].type.empty() )
//            outputs[ n.value ] = { type_name( std::get<n.value>( ret ) ), std::get<n.value>( ret ) };
//        else
//            ASSERT( std::get<n.value>( ret ) == nullptr, "" );
//    } );
//}

template<class T>
void Task::make_outputs( TypedVariant<T> *ret ) {
    set_output( Type{ type_name<T>() }, ret->data );
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
