#ifndef PAREX_VEC_H
#define PAREX_VEC_H

#include <asimd/AlignedAllocator.h>
#include <asimd/processing_units.h>
#include <functional>
#include <vector>

#include "../type_name.h"

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif // __CUDACC__

/**
  Aligned Vector that works (nearly) the same way in GPUs and CPUs
*/
template<class T,class A=asimd::Arch::Native,class Enable=void>
class Vec;

/**
  Specialization for cpus
*/
template<class T,class A>
class Vec<T,A,typename std::enable_if<A::cpu>::type> : public std::vector<T,asimd::AlignedAllocator<T,A>> {
public:
    using std::vector<T,asimd::AlignedAllocator<T,A>>::vector;
    using TI = typename A::size_t;

    template<class G,class B>
    Vec( const Vec<G,B> &v ) : std::vector<T,asimd::AlignedAllocator<T,A>>( v.size() ) { for( TI i = 0; i < v.size(); ++i ) this->operator[]( i ) = v[ i ]; }

    const T *ptr() const { return this->data(); }
    T *ptr() { return this->data(); }

    template<class Arg>
    void resize_and_set( TI new_size, Arg &&arg ) { this->clear(); this->resize( new_size, std::forward<Arg>( arg ) ); }

    static std::string type_name() { return "parex::Vec<" + type_name( S<T>() ) + ",parex::Arch::" + A::name() + ">"; }
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

#endif // PAREX_VEC_H
