#include "Pool.h"

template<class T,int buf_size>
Pool<T,buf_size>::Pool() : last_buf( new Buf ) {
    last_buf->free = last_buf->data;
    last_buf->prev = nullptr;
    last_free      = nullptr;
}

template<class T,int buf_size>
Pool<T,buf_size>::~Pool() {
    while ( last_buf ) {
        for( char *p = last_buf->free; ( p -= sizeof( T ) ) >= last_buf->data; )
            reinterpret_cast<T *>( p )->~T();
        Buf *prev = last_buf->prev;
        delete last_buf;
        last_buf = prev;
    }
}

template<class T,int buf_size> template  <class... Args>
T *Pool<T,buf_size>::New( Args &&...args ) {
    // we have a free item ?


    // we can use the last Buf ?
    Buf *old_last = last_buf;
    char *free = old_last->free, *next = free + sizeof( T );
    if ( next <= old_last->limit() ) {
        old_last->free = next;
        return new ( free ) T( std::forward<Args>( args )... );
    }

    // else, we have to create a new one
    last_buf = new Buf;
    last_buf->prev = old_last;
    last_buf->free = last_buf->data + sizeof( T );
    return new ( last_buf->data ) T( std::forward<Args>( args )... );

}

template<class T,int buf_size>
void Pool<T,buf_size>::free( T *ptr ) {

}
