#ifndef PAREX_TENSOR_H
#define PAREX_TENSOR_H

#include "Vec.h"

namespace parex {

/**
  Simple tensor class
*/
template<class T,class A=Arch::Native,class TI=std::size_t>
class Tensor {
public:
    /**/                 Tensor         ( const Vec<TI> &size, Vec<T> &&data ) : rese( size ), size( size ), data( std::forward<Vec<T>>( data ) ) {}
    template<class Data> Tensor         ( const Vec<TI> &size, Data &&data ) : rese( size ), size( size ), data( std::forward<Data>( data ) ) {}
    /**/                 Tensor         () {}

    void                 write_to_stream( std::ostream &os ) const;

    Vec<TI>              rese;
    Vec<TI>              size;
    Vec<T,A>             data;
};

} // namespace parex

#include "Tensor.tcc"

#endif // PAREX_TENSOR_H