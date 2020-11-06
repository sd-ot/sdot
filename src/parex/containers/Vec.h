#ifndef PAREX_VEC_H
#define PAREX_VEC_H

#include "../arch/AlignedAllocator.h"
#include "../arch/Arch.h"
#include <vector>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif // __CUDACC__

namespace parex {

/**
  Aligned Vector that works (nearly) the same way in GPUs and CPUs
*/
template<class T,class A=Arch::Native,class Enable=void>
class Vec;

/**
  Specialization for cpus
*/
template<class T,class A>
class Vec<T,A,typename std::enable_if<A::cpu>::type> : public std::vector<T,AlignedAllocator<T,A>> {
public:
    using std::vector<T,AlignedAllocator<T,A>>::vector;
};

#ifdef __CUDACC__

/**
  Specialization for gpus
*/
template<class T,class A>
class Vec<T,A,typename std::enable_if<A::gpu>::type> : thrust::device_vector<T> {
public:
    using thrust::device_vector<T>::device_vector;

    void write_to_stream( std::ostream &os ) const { thrust::host_vector<T> h = *this; os << h; }
};

#endif // __CUDACC__

} // namespace parex

#endif // PAREX_VEC_H
