#ifndef PAREX_TENSOR_H
#define PAREX_TENSOR_H

#include "../type_name.h"
#include <functional>
#include "Vec.h"

namespace parex {

/**
  Simple tensor class
*/
template<class T,class A=Arch::Native,class TI=std::size_t>
class Tensor {
public:
    /**/               Tensor         ( const Vec<TI> &size, Vec<T> &&data, const Vec<TI> &rese );
    /**/               Tensor         ( const Vec<TI> &size, Vec<T> &&data );
    /**/               Tensor         ( const Vec<TI> &size );
    /**/               Tensor         ();

    void               write_to_stream( std::ostream &os ) const;
    void               for_each_index ( const std::function<void( Vec<TI> &index, TI &off )> &f ) const;
    bool               next_index     ( Vec<TI> &index, TI &off ) const;
    TI                 init_mcum      ();
    TI                 nb_x_vec       () const { return mcum.back() / ( rese[ 0 ] + ( rese[ 0 ] == 0 ) ); }
    TI                 dim            () const { return size.size(); }

    void               resize         ( TI new_x_size );

    T*                 ptr            ( TI num_x ) { return ptr() + rese[ 0 ] * num_x; }
    const T*           ptr            ( TI num_x ) const { return ptr() + rese[ 0 ] * num_x; }

    const T*           ptr            () const { return data.data(); }
    T*                 ptr            () { return data.data(); }

    static std::string type_name      ();

    Vec<TI>            rese;
    Vec<TI>            mcum;
    Vec<TI>            size;
    Vec<T,A>           data;
};

} // namespace parex

#include "Tensor.tcc"

#endif // PAREX_TENSOR_H
