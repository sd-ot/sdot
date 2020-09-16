#ifndef SDOT_VEC_H
#define SDOT_VEC_H

#include "AlignedAllocator.h"
#include "MachineArch.h"
#include <numeric>
#include <vector>

namespace sdot {

/**
  Aligned Vector that works (nearly) the same way in GPUs and CPUs
*/
template<class T,class Arch=MachineArch::Native,int gpu = OnGpu<Arch>::value>
class Vec;

/**
  Specialization for cpus
*/
template<class T,class Arch>
class Vec<T,Arch,0> : public std::vector<T,AlignedAllocator<T,Arch>> {
public:
    using value_type = T;

    using std::vector<T,AlignedAllocator<T,Arch>>::vector;

    void  fill_iota      ( std::size_t b, std::size_t e, T beg ) { std::iota( this->begin() + b, this->begin() + e, beg ); }
    void  fill           ( std::size_t b, std::size_t e, T val ) { std::fill( this->begin() + b, this->begin() + e, val ); }

    const T* ptr         () const { return this->data(); }
    T*       ptr         () { return this->data(); }
};

} // namespace sdot

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>

namespace sdot {

/**
  Specialization for gpus
*/
template<class T,class Arch>
class Vec<T,Arch,1> : public thrust::device_vector<T> {
public:
    using    value_type = T;

    using    thrust::device_vector<T>::device_vector;

    //    template <class G,class B>
    //    /**/     Vec            ( const Vec<G,B,0> &that ) : thrust::device_vector<T>( that.begin(), that.end() ) {}

    void     write_to_stream( std::ostream &os ) const { thrust::host_vector<T> h = *this; os << h; }

    void     fill_iota      ( std::size_t b, std::size_t e, T beg ) { thrust::sequence( this->begin() + b, this->begin() + e, beg ); }
    void     fill           ( std::size_t b, std::size_t e, T val ) { thrust::fill    ( this->begin() + b, this->begin() + e, val ); }

    const T* ptr            () const { return thrust::raw_pointer_cast( this->data() ); }
    T*       ptr            () { return thrust::raw_pointer_cast( this->data() ); }
};

} // namespace sdot

#endif // __CUDACC__

#endif // SDOT_VEC_H
