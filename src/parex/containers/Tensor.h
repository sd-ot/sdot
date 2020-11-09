#ifndef PAREX_TENSOR_H
#define PAREX_TENSOR_H

#include "../support/Math.h"
#include <functional>
#include "Vec.h"

namespace parex {

/**
  Simple tensor class
*/
template<class T,class A=Arch::Native,class TI=std::size_t>
class Tensor {
public:
    template<class V>         Tensor         ( const V &size, Vec<T> &&data, const V &rese );
    template<class V>         Tensor         ( const V &size, Vec<T> &&data );
    template<class V>         Tensor         ( const V &size );
    /**/                      Tensor         () {}

    void                      write_to_stream( std::ostream &os ) const;
    void                      for_each_index ( const std::function<void( Vec<TI> &index, TI &off )> &f ) const;
    bool                      next_index     ( Vec<TI> &index, TI &off ) const;
    TI                        init_mcum      ();
    TI                        dim            () const { return size.size(); }

    void                      resize         ( TI new_x_size );

    Vec<TI>                   rese;
    Vec<TI>                   mcum;
    Vec<TI>                   size;
    Vec<T,A>                  data;
};

} // namespace parex

#include "Tensor.tcc"

#endif // PAREX_TENSOR_H
