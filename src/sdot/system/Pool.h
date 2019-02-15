#pragma once

#include <algorithm>

/**
*/
template<class T,int buf_size=1024>
class Pool {
public:
    /**/                       Pool        ();

    /**/                      ~Pool        ();

    template<class... Args> T *New         ( Args &&...args );
    void                       free        ( T *ptr );

private:
    struct    Buf {
        static constexpr int   footer_size = sizeof( Buf * ) + sizeof( char * );
        static constexpr int   data_size   = buf_size - footer_size;

        char                  *limit       () const { return (char *)&free; }

        char                   data[ data_size ]; ///<
        char                  *free;              ///< pointer to the last allocated item
        Buf                   *prev;              ///<
    };

    T*                         last_free;
    Buf*                       last_buf;
};

#include "Pool.tcc"
