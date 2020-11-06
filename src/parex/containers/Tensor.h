#ifndef PAREX_TENSOR_H
#define PAREX_TENSOR_H

#include <functional>
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

    void                 for_each_index ( const std::function<void( Vec<TI> &index, TI &off )> &f ) const;
    bool                 next_index     ( Vec<TI> &index, TI &off ) const;
    std::size_t          dim            () const { return size.size(); }

    void                 write_to_stream( std::ostream &os ) const;

    Vec<TI>              rese;
    Vec<TI>              size;
    Vec<T,A>             data;
};

} // namespace parex

#include "Tensor.tcc"

#endif // PAREX_TENSOR_H
