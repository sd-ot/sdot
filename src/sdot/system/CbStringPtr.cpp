#include "CbStringPtr.h"
#include <string.h>

namespace Hpipe {

CbStringPtr::CbStringPtr() : beg( 0 ), off( 0 ), end( 0 ) {
}

CbStringPtr::CbStringPtr( const CbStringPtr &bp ) : beg( bp.beg ), off( bp.off ), end( bp.end ) {
}

CbStringPtr::CbStringPtr( const CbString &bs ) {
    beg = bs.beg;
    off = bs.off;
    end = bs.end;
}

CbStringPtr::CbStringPtr( const CbQueue &bs ) {
    beg = bs.beg;
    off = bs.off;
    end = bs.off + bs.size();
}

CbStringPtr::CbStringPtr( const Buffer *beg_buf, const PI8 *beg_dat, const PI8 *end_dat ) {
    while ( beg_buf and ( beg_dat < beg_buf->data or beg_dat >= beg_buf->data + beg_buf->used ) )
        beg_buf = beg_buf->next;
    beg = beg_buf;

    if ( not beg_buf ) {
        off = 0;
        end = 0;
        return;
    }

    off = beg_dat - beg_buf->data;

    end = 0;
    while ( beg_buf and ( end_dat < beg_buf->data or end_dat >= beg_buf->data + beg_buf->used ) ) {
        end += beg_buf->used;
        beg_buf = beg_buf->next;
    }
    if ( beg_buf )
        end += end_dat - beg_buf->data;
}

CbStringPtr::CbStringPtr( const Buffer *beg_buff, const Buffer::PI8 *beg_data, const Buffer *end_buff, const Buffer::PI8 *end_data ) {
    beg = beg_buff;
    if ( beg_buff ) {
        off = beg_data - beg_buff->data;
        if ( end_buff == beg_buff ) {
            end = end_data - beg_buff->data;
        } else {
            end = 0;
            for ( ; beg_buff != end_buff; beg_buff = beg_buff->next )
                end += beg_buff->used;
            end += end_data - end_buff->data;
        }
    } else {
        off = 0;
        end = 0;
    }
}

bool CbStringPtr::ack_error() {
    off = -1;
    end = -1;
    return false;
}

void CbStringPtr::read_some( void *data, PT size ) {
    if ( not size )
        return;
    visitor( [ this, &data, &size ]( const Buffer *b, PT bd, PT ed ) {
        // last buffer to be used for this reading ?
        PT lm = ed - bd;
        if ( size <= lm ) {
            memcpy( data, b->data + bd, size );
            off += size;
            return false;
        }

        // else, read what we can
        size -= lm;
        memcpy( data, b->data + bd, lm );
        data = reinterpret_cast<PI8 *>( data ) + lm;

        // and remove b from the stream
        end -= b->used;
        beg = b->next;
        off = 0;
        return true;
    } );
}


Hpipe::CbStringPtr::operator std::string() const {
    std::string res; res.reserve( size() );

    data_visitor( [ &res ]( const PI8 *b, const PI8 *e ) {
        res.append( b, e );
    } );

    return res;
}



} // namespace Hpipe

