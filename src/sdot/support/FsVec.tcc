#include "FsVec.h"
#include <memory>

template<class T,class ST> template<class Pool>
FsVec<T,ST>::FsVec( Pool &pool, ST size, const T *content ) : _data( (T *)pool.allocate( size * sizeof( T ), alignof( T ) ) ), _size( size ) {
    for( ST i = 0; i < size; ++i )
        new ( _data + i ) T( content[ i ] );
}

template<class T,class ST> template<class Pool>
FsVec<T,ST>::FsVec( Pool &pool, ST size, ST alig ) : _data( (T *)pool.allocate( size * sizeof( T ), alig ) ), _size( size ) {
    for( ST i = 0; i < size; ++i )
        new ( _data + i ) T;
}

template<class T,class ST> template<class Pool>
FsVec<T,ST>::FsVec( Pool &pool, ST size ) : _data( (T *)pool.allocate( size * sizeof( T ), alignof( T ) ) ), _size( size ) {
    for( ST i = 0; i < size; ++i )
        new ( _data + i ) T;
}

template<class T,class ST>
FsVec<T,ST>::FsVec( T *data, ST size ) : _data( data ), _size( size ) {
}

template<class T,class ST>
FsVec<T,ST>::FsVec() : _data( nullptr ), _size( 0 ) {
}

template<class T,class ST>
const T &FsVec<T,ST>::operator[]( ST index ) const {
    return _data[ index ];
}

template<class T,class ST>
T &FsVec<T,ST>::operator[]( ST index ) {
    return _data[ index ];
}

template<class T,class ST>
ST FsVec<T,ST>::size() const {
    return _size;
}

template<class T,class ST>
const T* FsVec<T,ST>::begin() const {
    return _data;
}

template<class T,class ST>
T* FsVec<T,ST>::begin() {
    return _data;
}

template<class T,class ST>
const T* FsVec<T,ST>::data() const {
    return _data;
}

template<class T,class ST>
T* FsVec<T,ST>::data() {
    return _data;
}

template<class T,class ST>
const T* FsVec<T,ST>::end() const {
    return _data + _size;
}

template<class T,class ST>
T* FsVec<T,ST>::end() {
    return _data + _size;
}

template<class T,class ST>
const T &FsVec<T,ST>::front() const {
    return *_data;
}

template<class T,class ST>
T &FsVec<T,ST>::front() {
    return *_data;
}

template<class T,class ST>
void FsVec<T,ST>::downsize( ST size ) {
    _size = size;
}
