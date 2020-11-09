#include "../support/generic_ostream_output.h"
#include "../support/TODO.h"
#include "../support/P.h"

#include "../arch/assign_iota.h"
#include "../arch/copy.h"

#include "Tensor.h"

namespace parex {

template<class T,class A,class TI> template<class V>
Tensor<T,A,TI>::Tensor( const V &size, Vec<T> &&data, const V &rese ) : rese( rese ), size( size ), data( std::move( data ) ) {
    init_mcum();
}

template<class T,class A,class TI> template<class V>
Tensor<T,A,TI>::Tensor( const V &size, Vec<T> &&data ) : rese( size ), size( size ), data( std::move( data ) ) {
    init_mcum();
}

template<class T,class A,class TI> template<class V>
Tensor<T,A,TI>::Tensor( const V &size ) : rese( size ), size( size ) {
    if ( dim() ) {
        this->rese[ 0 ] = ceil( rese[ 0 ], SimdSize<T,A>::value );
        this->data.resize( init_mcum() );
    }
}


template<class T,class A,class TI>
void Tensor<T,A,TI>::write_to_stream( std::ostream &os ) const {
    //    os << "size:" << size;
    //    os << " rese:" << rese;
    //    os << " data:";
    for_each_index( [&]( const Vec<TI> &index, TI off ) {
        for( std::size_t n = 0; n < index.size(); ++n ) {
            if ( index[ n ] ) {
                if ( n )
                    while( n-- )
                        os << "\n";
                else
                    os << " ";
                break;
            }
        }

        os << data[ off ];
    } );
}

template<class T,class A,class TI>
bool Tensor<T,A,TI>::next_index( Vec<TI> &index, TI &off ) const {
    ++off;

    std::size_t n = 0;
    while ( true ) {
        if ( n == dim() )
            return false;
        if ( ++index[ n ] < size[ n ] )
            return true;
        index[ n++ ] = 0;
    }
}

template<class T, class A, class TI>
TI Tensor<T,A,TI>::init_mcum() {
    mcum.resize( dim() + 1 );

    mcum[ 0 ] = 1;
    for( std::size_t i = 0, m = 1; i < dim(); ++i ) {
        m *= this->rese[ i ];
        mcum[ i + 1 ] = m;
    }

    return mcum.back();
}

template<class T,class A,class TI>
void Tensor<T,A,TI>::for_each_index( const std::function<void(Vec<TI>&,TI &off)> &f ) const {
    if ( dim() == 0 || mcum[ dim() ] == 0 )
        return;

    Vec<TI> index( dim(), 0 );
    index[ 0 ] = TI( -1 );
    TI off = TI( -1 );
    while ( next_index( index, off ) )
        f( index, off );
}

template<class T,class A,class TI>
void Tensor<T,A,TI>::resize( TI new_x_size ) {
    if ( dim() == 0 || rese[ 0 ] >= new_x_size )
        return;

    TI old_x_rese = std::exchange( rese[ 0 ], ceil( new_x_size, SimdSize<T,A>::value ) );
    TI old_len_xs = mcum.back() ? mcum.back() / old_x_rese : 0;
    TI old_x_size = std::exchange( size[ 0 ], new_x_size );

    Vec<T,A> old_data = std::move( data );
    data.resize( init_mcum() );

    for( TI i = 0; i < old_len_xs; ++i )
        copy( data.data() + rese[ 0 ] * i, old_data.data() + old_x_rese * i, old_x_size );
}

} // namespace parex
