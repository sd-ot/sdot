#pragma once

#include <stdlib.h>
#include <new>

namespace Hpipe {

/**
*/
class Buffer {
public:
    using CBF = const Buffer *;
    using PI8 = unsigned char;
    using PT  = size_t;

    enum { default_size = 2048 - 2 * sizeof( PT ) - sizeof( Buffer * ) - sizeof( int ), nb_in_base_data = 4 };

    Buffer( PT used, PT size, Buffer *prev = 0 ) : used( used ), size( size ), next( 0 ), cpt_use( 0 ) {
        if ( prev )
            prev->next = this;
    }

    static Buffer *New( PT size = default_size, Buffer *prev = 0 ) {
        #ifdef HPIPE_CHECK_ALIVE_BUF
        ++nb_alive_bufs;
        #endif // HPIPE_CHECK_ALIVE_BUF

        // update size for alignment
        size = ( size + sizeof( PT ) - 1 ) & ~( sizeof( PT ) - 1 );
        return new ( malloc( sizeof( Buffer ) - 4 + size ) ) Buffer( 0, size, prev );
    }

    static void Free( Buffer *buf ) {
        #ifdef HPIPE_CHECK_ALIVE_BUF
        --nb_alive_bufs;
        #endif // HPIPE_CHECK_ALIVE_BUF
        free( (Buffer *)buf );
    }

    /// return nb bytes in skipped buffers
    static PT skip( CBF *buf, const PI8 *&data, PT nb_to_skip ) {
        if ( ! *buf || ! nb_to_skip )
            return false;

        PT skipped = 0;
        while ( nb_to_skip >= PT( (*buf)->data + (*buf)->used - data ) ) {
            nb_to_skip -= (*buf)->data + (*buf)->used - data;
            const Buffer *old = *buf;
            skipped += (*buf)->used;
            *buf = (*buf)->next;
            dec_ref( old );
            if ( ! *buf )
                return skipped;
            data = (*buf)->data;
        }
        data += nb_to_skip;
        return skipped;
    }

    /// return true if buf has changed
    static PT skip( CBF *buf, const PI8 *&data, PT nb_to_skip, unsigned nb_to_keep ) {
        if ( ! *buf )
            return false;

        PT skipped = 0;
        while ( nb_to_skip >= PT( (*buf)->data + (*buf)->used - data ) ) {
            nb_to_skip -= (*buf)->data + (*buf)->used - data;
            const Buffer *old = *buf;
            skipped += (*buf)->used;
            *buf = (*buf)->next;
            if ( nb_to_keep )
                inc_ref( old, nb_to_keep - 1 );
            else
                dec_ref( old );
            if ( ! *buf )
                return skipped;
            data = (*buf)->data;
        }
        data += nb_to_skip;
        return skipped;
    }

    static PT size_between( const Buffer *beg_buf, const PI8 *beg_data, const Buffer *end_buf, const PI8 *end_data ) {
        if ( beg_buf == end_buf )
            return end_data - beg_data;

        PT res = beg_buf->data + beg_buf->used - beg_data;
        while ( true ) {
            beg_buf = beg_buf->next;
            if ( beg_buf == end_buf ) {
                res += end_data - end_buf->data;
                return res;
            }
            res += beg_buf->used;
        }
    }

    static const Buffer *inc_ref( const Buffer *p ) {
        ++p->cpt_use;
        return p;
    }

    static Buffer *inc_ref( Buffer *p ) {
        ++p->cpt_use;
        return p;
    }

    static const Buffer *inc_ref( const Buffer *p, unsigned n ) {
        p->cpt_use += n;
        return p;
    }

    static Buffer *inc_ref( Buffer *p, unsigned n ) {
        p->cpt_use += n;
        return p;
    }

    static void dec_ref( const Buffer *p ) {
        if ( --p->cpt_use < 0 )
            Free( (Buffer *)p );
    }

    PT room() const {
        return size - used;
    }

    PT cum_size() const {
       PT res = 0;
       for( const Buffer *b = this; b; b = b->next )
           res += b->used;
       return res;
    }

    void dec_ref_rec() {
        for( Buffer *buf = this; buf; ) {
            Buffer *old = buf;
            buf = buf->next;
            dec_ref( old );
        }
    }

    void dec_ref_upto( Buffer *dst ) {
        for( Buffer *buf = this; buf != dst; ) {
            Buffer *old = buf;
            buf = buf->next;
            dec_ref( old );
        }
    }

    /// nb_to_keep = 0 <=> dec_ref_upto( dst ) ()
    void dec_ref_upto( Buffer *dst, unsigned nb_to_keep ) {
        for( Buffer *buf = this; buf != dst; ) {
            Buffer *old = buf;
            buf = buf->next;
            if ( nb_to_keep )
                inc_ref( old, nb_to_keep - 1 );
            else
                dec_ref( old );
        }
    }

    const PI8 *begin() const {
        return data;
    }

    PI8 *begin() {
        return data;
    }

    const PI8 *end() const {
        return data + used;
    }

    PI8 *end() {
        return data + used;
    }

    // attributes
    PT          used;                    ///< nb items stored in data
    PT          size;                    ///< real size of data[]
    Buffer     *next;                    ///<
    mutable int cpt_use;                 ///< destroyed if < 0
    PI8         data[ nb_in_base_data ]; ///<
    #ifdef HPIPE_CHECK_ALIVE_BUF
    static int  nb_alive_bufs;
    #endif // HPIPE_CHECK_ALIVE_BUF
};

}
