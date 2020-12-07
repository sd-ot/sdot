#pragma once

#include "../containers/gtensor.h"
#include <initializer_list>
#include "Scalar.h"

namespace parex {

/**
  A wrapper around a `Task`, with constructors and operators for tensors
*/
class Tensor : public TaskWrapper {
public:
    template<class T> Tensor    ( std::initializer_list<std::initializer_list<std::initializer_list<T>>> &&l );
    template<class T> Tensor    ( std::initializer_list<std::initializer_list<T>> &&l );
    template<class T> Tensor    ( std::initializer_list<T> &&l );
    /**/              Tensor    ( Task *t );
    /**/              Tensor    ();

    Tensor            operator+ ( const Tensor &that ) const;
    Tensor            operator- ( const Tensor &that ) const;
    Tensor            operator* ( const Tensor &that ) const;
    Tensor            operator/ ( const Tensor &that ) const;

    Tensor&           operator+=( const Tensor &that );
    Tensor&           operator-=( const Tensor &that );
    Tensor&           operator*=( const Tensor &that );
    Tensor&           operator/=( const Tensor &that );
};

template<class T>
Tensor::Tensor( std::initializer_list<std::initializer_list<std::initializer_list<T>>> &&l ) {
    task = SrcTask::from_ptr( new gtensor<T,3,CpuAllocator>( &CpuAllocator::local, std::move( l ) ), /*owned*/ true );
}

template<class T>
Tensor::Tensor( std::initializer_list<std::initializer_list<T>> &&l ) {
    task = SrcTask::from_ptr( new gtensor<T,2,CpuAllocator>( &CpuAllocator::local, std::move( l ) ), /*owned*/ true );
}

template<class T>
Tensor::Tensor( std::initializer_list<T> &&l ) {
    task = SrcTask::from_ptr( new gtensor<T,1,CpuAllocator>( &CpuAllocator::local, std::move( l ) ), /*owned*/ true );
}

} // namespace parex
