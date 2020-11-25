#ifndef PAREX_VEC_H
#define PAREX_VEC_H

#include "../arch/AlignedAllocator.h"
#include "../arch/Arch.h"
#include "../type_name.h"
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

    template<class G,class B>
    Vec( const Vec<G,B> &v ) : std::vector<T,AlignedAllocator<T,A>>( v.size() ) { for( std::size_t i = 0; i < v.size(); ++i ) this->operator[]( i ) = v[ i ]; }

    const T *ptr() const { return this->data(); }
    T *ptr() { return this->data(); }

    template<class Arg>
    void resize_and_set( std::size_t new_size, Arg &&arg ) { this->clear(); this->resize( new_size, std::forward<Arg>( arg ) ); }

    static std::string type_name() { return "parex::Vec<" + parex::type_name<T>() + ",parex::Arch::" + A::name() + ">"; }
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