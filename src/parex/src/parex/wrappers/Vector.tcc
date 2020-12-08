#include "../resources/default_CpuAllocator.h"
#include "../containers/gvector.h"
#include "Vector.h"

namespace parex {

template<class T>
Vector::Vector( std::initializer_list<T> &&l ) {
    task = Task::new_src_from_ptr( new gvector<T,BasicCpuAllocator>( &default_CpuAllocator, std::move( l ) ), /*owned*/ true );
}

} // namespace parex
