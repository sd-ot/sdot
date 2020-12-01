#include "gtensor.h"
#include "../Math.h"
#include "../TODO.h"

template<class T,std::size_t N,class A>
gtensor<T,N,A>::gtensor() {
    for( std::size_t i = 0; i < N; ++i ) {
        _size[ i ] = 0;
        _rese[ i ] = 0;
        _cpre[ i ] = 0;
    }
}

template<class T,std::size_t N,class A>
gtensor<T,N,A>::~gtensor() {
}

template<class T,std::size_t N,class A> template<class... Args>
void gtensor<T,N,A>::resize( A &allocator, Args&& ...args ) {
    S  old_shape = _size;
    S  old_rese = _rese;
    S  old_cpre = _cpre;
    T *old_data = _data;

    _size = { I( args )... };
    _rese = { I( args )... };
    if ( I a = A::alignment / sizeof( T ) )
        _rese[ N - 1 ] = ceil( _rese[ N - 1 ], a );
    _update_cpre();
    _data = allocator.allocate( sizeof( T ) * _cpre[ N - 1 ] );

    if ( old_cpre[ N - 1 ] ) {
        TODO;
        allocator.deallocate( old_data );
    }
}

template<class T,std::size_t N,class A>
void gtensor<T,N,A>::write_to_stream( std::ostream &os, const A &allocator ) const {
    auto add_spaces = []( std::ostream &os, std::array<I,N> inds ) {
        if ( inds[ N - 1 ] ) {
            os << " ";
            return;
        }

        for( int n = N - 1; ; ) {
            if ( n-- == 0 )
                return;
            if ( inds[ n ] ) {
                while( ++n < N )
                    os << "\n";
                return;
            }
        }
    };

    for_each_index( [&]( auto... ind ) {
        add_spaces( os, { ind... } );
        os << at( allocator, ind... );
    } );
}

template<class T,std::size_t D,class A> template<class F,int d,class...Z>
void gtensor<T,D,A>::_for_each_index( F &&f, N<d>, Z&& ...inds ) const {
    for( std::size_t i = 0; i < _size[ d ]; ++i )
        _for_each_index( f, N<d+1>(), inds..., i );
}

template<class T,std::size_t D,class A> template<class F,class...Z>
void gtensor<T,D,A>::_for_each_index( F &&f, N<D>, Z&& ...inds ) const {
    f( inds... );
}

template<class T,std::size_t D,class A> template<class F>
void gtensor<T,D,A>::for_each_index( F &&f ) const {
    _for_each_index( f, N<0>() );
}

template<class T,std::size_t N,class A> template<class... Args>
T gtensor<T,N,A>::at( const A &allocator, Args&& ...args ) const {
    return allocator.value( _data + index( std::forward<Args>( args )... ) );
}

template<class T,std::size_t N,class A> template<class... Args>
typename gtensor<T,N,A>::I gtensor<T,N,A>::index( Args&& ...args ) const {
    return _mul_cpre( 1, args... );
}

template<class T,std::size_t N,class A>
void gtensor<T,N,A>::_update_cpre() {
    for( std::size_t i = N, c = 1; i--; )
        _cpre[ i ] = ( c *= _rese[ i ] );
}
