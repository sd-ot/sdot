#include <asimd/operations/assign_iota.h>
#include <asimd/operations/copy.h>
#include <asimd/SimdVec.h>

#include "../generic_ostream_output.h"
#include "../Math.h"
#include "../TODO.h"

#include "Tensor.h"

template<class T,class A>
Tensor<T,A>::Tensor( const Vec<TI> &size, Vec<T> &&data, const Vec<TI> &rese ) : rese( rese ), size( size ), data( std::move( data ) ) {
    init_mcum();
}

template<class T,class A>
Tensor<T,A>::Tensor( const Vec<TI> &size, Vec<T> &&data ) : rese( size ), size( size ), data( std::move( data ) ) {
    init_mcum();
}

template<class T,class A>
Tensor<T,A>::Tensor( const Vec<TI> &size ) : rese( size ), size( size ) {
    if ( dim() ) {
        this->rese[ 0 ] = ceil( rese[ 0 ], asimd::SimdSize<T,A>::value );
        this->data.resize( init_mcum() );
    }
}

template<class T,class A>
Tensor<T,A>::Tensor() : mcum( 0 ) {
}

template<class T,class A>
void Tensor<T,A>::write_to_stream( std::ostream &os ) const {
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

template<class T,class A>
bool Tensor<T,A>::next_index( Vec<TI> &index, TI &off ) const {
    ++off;

    // x
    if ( ++index[ 0 ] < size[ 0 ] ) {
        return true;
    }

    off += rese[ 0 ] - size[ 0 ];
    index[ 0 ] = 0;

    //
    for( std::size_t n = 1; n < dim(); ++n ) {
        if ( ++index[ n ] < size[ n ] )
            return true;
        index[ n ] = 0;
    }
    return false;
}

template<class T,class A>
typename Tensor<T,A>::TI Tensor<T,A>::init_mcum() {
    mcum.resize( dim() + 1 );

    mcum[ 0 ] = 1;
    for( std::size_t i = 0, m = 1; i < dim(); ++i ) {
        m *= this->rese[ i ];
        mcum[ i + 1 ] = m;
    }

    return mcum.back();
}

template<class T,class A>
void Tensor<T,A>::for_each_index( const std::function<void(Vec<TI>&,TI &off)> &f ) const {
    if ( mcum[ dim() ] == 0 )
        return;

    Vec<TI> index( dim(), 0 );
    index[ 0 ] = TI( -1 );
    TI off = TI( -1 );
    while ( next_index( index, off ) )
        f( index, off );
}

template<class T,class A>
void Tensor<T,A>::resize( TI new_x_size ) {
    if ( dim() == 0 || rese[ 0 ] >= new_x_size )
        return;

    TI old_x_rese = std::exchange( rese[ 0 ], ceil( new_x_size, asimd::SimdSize<T,A>::value ) );
    TI old_len_xs = mcum.back() ? mcum.back() / old_x_rese : 0;
    TI old_x_size = std::exchange( size[ 0 ], new_x_size );

    Vec<T,A> old_data = std::move( data );
    data.resize( init_mcum() );

    for( TI i = 0; i < old_len_xs; ++i )
        copy( data.data() + rese[ 0 ] * i, old_data.data() + old_x_rese * i, old_x_size );
}

template<class T,class A>
std::string Tensor<T,A>::type_name() {
    return "parex::Tensor<" + type_name( S<T>() ) + ",parex::Arch::" + A::name() + ">";
}
