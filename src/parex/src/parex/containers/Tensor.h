#ifndef PAREX_TENSOR_H
#define PAREX_TENSOR_H

#include "Vec.h"

/**
  Simple tensor class
*/
template<class T,class A=asimd::Arch::Native>
class Tensor {
public:
    using              TI             = typename A::TI;

    /**/               Tensor         ( const Vec<TI> &size, Vec<T> &&data, const Vec<TI> &rese );
    /**/               Tensor         ( const Vec<TI> &size, Vec<T> &&data );
    /**/               Tensor         ( const Vec<TI> &size );
    /**/               Tensor         ();

    void               write_to_stream( std::ostream &os ) const;
    void               for_each_index ( const std::function<void( Vec<TI> &index, TI &off )> &f ) const;
    bool               next_index     ( Vec<TI> &index, TI &off ) const;
    TI                 init_mcum      ();
    TI                 nb_x_vec       () const { return mcum.back() / ( rese[ 0 ] + ( rese[ 0 ] == 0 ) ); }
    TI                 x_size         () const { return size[ 0 ]; }
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

#include "Tensor.tcc"

#endif // PAREX_TENSOR_H