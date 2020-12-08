#include "../tasks/CompiledTask.h"
#include "../plugins/Src.h"
#include "Tensor.h"

namespace parex {

template<class T>
Tensor::Tensor( std::initializer_list<std::initializer_list<std::initializer_list<T>>> &&l ) {
    task = Task::new_src_from_ptr( new gtensor<T,3,BasicCpuAllocator>( &default_CpuAllocator, std::move( l ) ), /*owned*/ true );
}

template<class T>
Tensor::Tensor( std::initializer_list<std::initializer_list<T>> &&l ) {
    task = Task::new_src_from_ptr( new gtensor<T,2,BasicCpuAllocator>( &default_CpuAllocator, std::move( l ) ), /*owned*/ true );
}

template<class T>
Tensor::Tensor( std::initializer_list<T> &&l ) {
    task = Task::new_src_from_ptr( new gtensor<T,1,BasicCpuAllocator>( &default_CpuAllocator, std::move( l ) ), /*owned*/ true );
}

} // namespace parex
