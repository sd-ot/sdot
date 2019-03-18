#pragma once

#include <cstdint>

/**
*/
template<class T>
class Span {
public:
    /*  */                     Span      ( const T *begin, std::size_t size ) : _data( begin ), _size( size ) { }
    /*  */                     Span      ( const T *begin, const T *end ) : _data( begin ), _size( end - begin ) { }

    const T*                   begin     () const { return _data; }
    const T*                   end       () const { return _data + _size; }

    auto                       size      () const { return _size; }

    template<class I> const T &operator[]( I i ) const { return _data[ i ]; }

private:
    const T*                   _data;
    std::size_t                _size;
};
