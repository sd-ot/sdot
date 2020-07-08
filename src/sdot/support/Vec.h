#ifndef SDOT_VEC_H
#define SDOT_VEC_H

#include "AlignedAllocator.h"
#include "CpuArch.h"
#include <vector>

/**
  Aligned Vector that works (nearly) the same way in GPUs and CPUs
*/
template<class T,class Arch=CpuArch::Native,class Enable=void>
class Vec;

/**
  Specialization for cpus
*/
template<class T,class Arch>
class Vec<T,Arch,typename std::enable_if<Arch::cpu>::type> : public std::vector<T,AlignedAllocator<T,Arch>> {
public:
    using std::vector<T,AlignedAllocator<T,Arch>>::vector;
};

#ifdef __CUDACC__
#include "thrust/device_vector.h"

/**
  Specialization for gpus
*/
template<class T,class Arch>
class Vec<T,Arch,typename std::enable_if<Arch::gpu>::type> : thrust::device_vector<T> {
public:
    using thrust::device_vector<T>::device_vector;

    void write_to_stream( std::ostream &os ) const { thrust::host_vector<T> h = *this; os << h; }
};

#endif // __CUDACC__

#endif // SDOT_VEC_H
