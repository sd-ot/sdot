#pragma once

#include "Math.h"

/**
  MemoryPool destroyed once (at the end). Destruction of items are not handled here.
*/
class IncrementalMemoryPool {
public:
    IncrementalMemoryPool() {
        last = new Buf;
        last->free = last->data;
        last->prev = 0;
    }

    ~IncrementalMemoryPool() {
        while ( last ) {
            Buf *prev = last->prev;
            delete last;
            last = prev;
        }
    }

    template<class T>
    T *create() {
        // we can use the last Buf ?
        auto free = reinterpret_cast<char *>( ceil( reinterpret_cast<std::size_t>( last->free ), alignof( T ) ) );
        auto next = free + sizeof( T );
        if ( next - last->data <= last->data_size ) {
            last->free = next;
            return reinterpret_cast<T *>( free );
        }

        // else, we have to create a new one
        Buf *new_buf = new Buf;
        free = reinterpret_cast<char *>( ceil( reinterpret_cast<std::size_t>( new_buf->data ), alignof( T ) ) );

        new_buf->free = free + sizeof( T );
        new_buf->prev = last;
        last = new_buf;

        return reinterpret_cast<T *>( free );
    }

private:
    enum {
        buf_size = 4096
    };

    struct Buf {
        enum {
            header_size = sizeof( Buf * ) + sizeof( int ),
            data_size   = buf_size - header_size
        };

        Buf  *prev;              ///<
        char *free;              ///<
        char  data[ data_size ]; ///<
    };

    Buf *last;
};

