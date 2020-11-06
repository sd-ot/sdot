#include "../support/generic_ostream_output.h"
#include "Tensor.h"

namespace parex {

template<class T,class A,class TI>
void Tensor<T,A,TI>::write_to_stream( std::ostream &os ) const {
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

template<class T,class A,class TI>
void Tensor<T,A,TI>::for_each_index( const std::function<void(Vec<TI>&,TI &off)> &f ) const {
    if ( ! dim() )
        return;

    Vec<TI> index( dim(), 0 );
    index[ 0 ] = TI( -1 );
    TI off = TI( -1 );
    while ( next_index( index, off ) )
        f( index, off );
}


} // namespace parex
