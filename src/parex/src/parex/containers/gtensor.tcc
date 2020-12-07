#include "../utility/ASSERT.h"
#include "../utility/TODO.h"
#include "../utility/Math.h"
#include "gtensor.h"

namespace parex {

template<class T,int N,class Allocator>
gtensor<T,N,Allocator>::gtensor( Allocator *allocator, S size, S rese, T *data, bool own ) : _allocator( allocator ), _size( size ), _rese( rese ), _data( data ), _own( own ) {
    _update_cprs();
}

template<class T,int N,class Allocator>
gtensor<T,N,Allocator>::gtensor( Allocator *allocator, S size, T *data, bool own ) : gtensor( allocator, size, size, data, own ) {
}

template<class T,int N,class A> template<class U>
gtensor<T,N,A>::gtensor( A *allocator, std::initializer_list<std::initializer_list<std::initializer_list<U>>> &&l ) : _allocator( allocator ), _own( true ) {
    static_assert( N == 3, "3 level initializer_list => dim = 3" );
    if ( l.size() ) {
        _size = { l.size(), l.begin()->size(), l.begin()->begin()->size() };
        _rese = _size;
        _update_rese();
        _update_cprs();
        _data = allocator->template allocate<T>( _cprs[ 0 ] );

        I o = 0;
        for( auto &s : l ) {
            for( auto &t : s ) {
                for( auto &v : t )
                    _set_at( o++, std::move( v ) );
                o += _rese[ N - 1 ] - _size[ N - 1 ];
            }
        }
    } else {
        _size = _null_S();
        _rese = _null_S();
        _cprs = _null_S();
        _data = nullptr;
    }
}

template<class T,int N,class A> template<class U>
gtensor<T,N,A>::gtensor( A *allocator, std::initializer_list<std::initializer_list<U>> &&l ) : _allocator( allocator ), _own( true ) {
    static_assert( N == 2, "2 level initializer_list => dim = 2" );
    if ( l.size() ) {
        _size = { l.size(), l.begin()->size() };
        _rese = _size;
        _update_rese();
        _update_cprs();
        _data = allocator->template allocate<T>( _cprs[ 0 ] );

        I o = 0;
        for( auto &s : l ) {
            for( auto &v : s )
                _set_at( o++, std::move( v ) );
            o += _rese[ N - 1 ] - _size[ N - 1 ];
        }
    } else {
        _size = _null_S();
        _rese = _null_S();
        _cprs = _null_S();
        _data = nullptr;
    }
}

template<class T,int N,class A> template<class U>
gtensor<T,N,A>::gtensor( A *allocator, std::initializer_list<U> &&l ) {
    static_assert( N == 1, "1 level initializer_list => dim = 1" );
    if ( l.size() ) {
        _size = { l.size() };
        _rese = _size;
        _update_rese();
        _update_cprs();
        _data = allocator->template allocate<T>( _cprs[ 0 ] );

        I o = 0;
        for( auto &v : l )
            _set_at( o++, std::move( v ) );
    } else {
        _size = _null_S();
        _rese = _null_S();
        _cprs = _null_S();
        _data = nullptr;
    }
}

template<class T,int N,class A>
gtensor<T,N,A>::gtensor( gtensor &&that ) : _allocator( that._allocator ), _size( that._size ), _rese( that._rese ), _cprs( that._cprs ), _data( that._data ), _own( that._own ) {
    that._clear();
}

template<class T,int N,class A>
gtensor<T,N,A>::gtensor( const gtensor &that ) : _allocator( that._allocator ), _size( that._size ), _rese( that._rese ), _cprs( that._cprs ), _own( true ) {
    _allocate();
    TODO; // copy of data
}

template<class T,int N,class A>
gtensor<T,N,A>::~gtensor() {
    if ( _own && _allocator && _cprs[ 0 ] )
        _allocator->deallocate( _data, _cprs[ 0 ] );
}

template<class T,int N,class A> template<class... Args>
void gtensor<T,N,A>::resize( Args&& ...args ) {
    S  old_cprs = _cprs;
    T *old_data = _data;

    _size = { I( args )... };
    _rese = { I( args )... };
    _update_rese();
    _update_cprs();
    _allocate();

    if ( old_cprs[ 0 ] ) {
        TODO; // copy of content
        _allocator->deallocate( old_data );
    }
}

template<class T,int N,class A>
void gtensor<T,N,A>::_clear() {
    _allocator = nullptr;
    _size = _null_S();
    _rese = _null_S();
    _cprs = _null_S();
    _data = nullptr;
}

template<class T,int N,class A>
void gtensor<T,N,A>::write_to_stream( std::ostream &os ) const {
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

    // max entry width
    I max_width = 0;
    for_each_index( [&]( auto... ind ) {
        std::ostringstream ss;
        ss << at( ind... );
        max_width = std::max( max_width, ss.str().size() );
    } );

    //
    for_each_index( [&]( auto... ind ) {
        std::ostringstream ss;
        ss << at( ind... );

        add_spaces( os, { ind... } );
        os << std::string( max_width - ss.str().size(), ' ' );
        os << ss.str();
    } );
}

template<class T,int D,class A> template<class F,int d,class...Z>
void gtensor<T,D,A>::_for_each_offset_and_index( F &&f, N<d>, I off, Z&& ...inds ) const {
    for( std::size_t i = 0; i < _size[ d ]; ++i )
        _for_each_offset_and_index( f, N<d+1>(), off + _cprs[ d + 1 ] * i, inds..., i );
}

template<class T,int D,class A> template<class F,class...Z>
void gtensor<T,D,A>::_for_each_offset_and_index( F &&f, N<D-1>, I off, Z&& ...inds ) const {
    if ( D )
        for( std::size_t i = 0; i < _size[ D - 1 ]; ++i )
            f( off++, inds..., i );
}

template<class T,int D,class A> template<class F>
void gtensor<T,D,A>::for_each_offset_and_index( F &&f ) const {
    _for_each_offset_and_index( f, N<0>(), 0 );
}


template<class T,int D,class A> template<class F,int d,class...Z>
void gtensor<T,D,A>::_for_each_index( F &&f, N<d>, Z&& ...inds ) const {
    for( std::size_t i = 0; i < _size[ d ]; ++i )
        _for_each_index( f, N<d+1>(), inds..., i );
}

template<class T,int D,class A> template<class F,class...Z>
void gtensor<T,D,A>::_for_each_index( F &&f, N<D>, Z&& ...inds ) const {
    f( inds... );
}

template<class T,int D,class A> template<class F>
void gtensor<T,D,A>::for_each_index( F &&f ) const {
    _for_each_index( f, N<0>() );
}

template<class T,int N,class A> template<class... Args>
T gtensor<T,N,A>::at( Args&& ...args ) const {
    return _get_at( offset( std::forward<Args>( args )... ) );
}

template<class T,int N,class A> template<class... Args>
typename gtensor<T,N,A>::I gtensor<T,N,A>::offset( Args&& ...args ) const {
    return _mul_cprs( 1, args... );
}

template<class T,int N,class A>
void gtensor<T,N,A>::_update_cprs() {
    for( std::size_t i = N, c = 1; i--; )
        _cprs[ i ] = ( c *= _rese[ i ] );
}

template<class T,int N,class A>
void gtensor<T,N,A>::_update_rese() {
    _rese[ N - 1 ] = ceil( _rese[ N - 1 ], A::template Alignment<T>::value );
}

template<class T,int N,class A>
void gtensor<T,N,A>::_allocate() {
    _data = _allocator->template allocate<T>( _cprs[ 0 ] );
}

template<class T,int N,class A>
typename gtensor<T,N,A>::S gtensor<T,N,A>::_null_S() {
    S res;
    for( std::size_t i = 0; i < N; ++i )
        res[ i ] = 0;
    return res;
}

template<class T,int N,class A>
void gtensor<T,N,A>::_set_at( I index, const T &value ) {
    copy_memory_values( *_allocator, _data + index, CpuAllocator(), &value, 1 );
}

template<class T,int N,class A>
T gtensor<T,N,A>::_get_at( I index ) const {
    T value;
    copy_memory_values( CpuAllocator(), &value, *_allocator, _data + index, 1 );
    return value;
}

} // namespace parex
