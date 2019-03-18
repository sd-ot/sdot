#pragma once

#include "CbString.h"

namespace Hpipe {

/**
  Part of a buffer sequence. Read-only.

  Can be used as an input stream.

  Ref counting for buffer is not handled here.
*/
class CbStringPtr {
public:
    CbStringPtr();
    CbStringPtr( const CbStringPtr &bp );
    CbStringPtr( const CbString    &bs );
    CbStringPtr( const CbQueue     &bs );
    CbStringPtr( const Buffer *beg_buf, const PI8 *beg_dat, const PI8 *end_dat );
    CbStringPtr( const Hpipe::Buffer *beg_buff, const Buffer::PI8 *beg_data, const Hpipe::Buffer *end_buff, const Buffer::PI8 *end_data );

    // errors
    bool error() const { return ST( off ) < 0; }
    bool ack_error();

    // size
    bool empty() const { return end == off; }
    PT   size() const { return end - off; }

    const Buffer *get_buf() { return beg; }
    PT            get_off() { return off; }
    PT            get_end() { return end; }

    // readers. Beware there are no checks in these methods (ack_read_... must be called before, to check if possible)
    void read_some( void *data, PT size );
    CbStringPtr &skip_some( PT size ) {
        visitor( [ this, &size ]( const Buffer *b, PT bd, PT ed ) {
            // last buffer to be used for this reading ?
            PT lm = ed - bd;
            if ( size <= lm ) {
                off += size;
                return false;
            }

            // else, read what we can
            size -= lm;

            // and remove b from the stream
            end -= b->used;
            beg = b->next;
            off = 0;
            return true;
        } );
        return *this;
    }

    PI8 read_byte() { return read_byte_wo_dec_ref( beg, off, end ); }

    // checkings for readers that save a signal (that will give error() != 0) if not ok. To be done before each read.
    bool ack_read_byte()         { if ( off   >=    end ) return ack_error(); return true; } ///< return true if ok to read a byte. Else, set end to 0 (to signal an error) and return false.
    bool ack_read_some( PT len ) { if ( off + len > end ) return ack_error(); return true; } ///< return true if ok to `len` bytes. Else, set end to 0 (to signal an error) and return false.

    // display
    operator std::string() const;

    explicit operator bool() const { return not empty(); }

    //
    void data_visitor( std::function<void(const PI8 *,const PI8 *)> func ) const { ///< op must return true to continue to read
        visitor( [ &func ]( const Buffer *b, ST bb, ST eb ) {
            func( b->data + bb, b->data + eb );
            return true;
        } );
    }

    //
    template<class Op>
    bool find_utf8( const Op &op ) const {
        const Buffer *l_beg = beg;
        PT l_off = off;
        PT l_end = end;
        while ( l_off < l_end ) {
            // first byte
            unsigned c0 = read_byte_wo_dec_ref( l_beg, l_off, l_end );

            // cases
            if ( c0 <= 0x7F ) { // 1 byte
                if ( op( c0 ) )
                    return true;
            } else if ( c0 <= 0xDF ) { // 2 bytes
                if ( l_off >= l_end - 0 )
                    return false;
                unsigned c1 = read_byte_wo_dec_ref( l_beg, l_off, l_end );
                if ( op( ( c0 << 8 ) + c1 ) )
                    return true;
            } else if ( c0 <= 0xEF ) { // 3 bytes
                if ( l_off >= l_end - 1 )
                    return false;
                unsigned c1 = read_byte_wo_dec_ref( l_beg, l_off, l_end );
                unsigned c2 = read_byte_wo_dec_ref( l_beg, l_off, l_end );
                if ( op( ( c0 << 16 ) + ( c1 << 8 ) + c2 ) )
                    return true;
            } else { // 4 bytes
                if ( l_off >= l_end - 2 )
                    return false;
                unsigned c1 = read_byte_wo_dec_ref( l_beg, l_off, l_end );
                unsigned c2 = read_byte_wo_dec_ref( l_beg, l_off, l_end );
                unsigned c3 = read_byte_wo_dec_ref( l_beg, l_off, l_end );
                if ( op( ( c0 << 24 ) + ( c1 << 16 ) + ( c2 << 8 ) + c3 ) )
                    return true;
            }
        }
        return false;
    }

    // display
    void write_to_stream( std::ostream &os ) const {
        data_visitor( [ &os ]( const PI8 *b, const PI8 *e ) {
            os.write( (const char *)b, e - b );
        } );
        //    visitor( [ &os ]( const Buffer *b, PT beg, PT end ) {
        //        static const char *c = "0123456789abcdef";
        //        for( PT i = beg; i < end; ++i )
        //            os << c[ b->data[ i ] / 16 ] << c[ b->data[ i ] % 16 ] << " ";
        //        return true;
        //    } );
    }

    template<class Op>
    bool visitor( const Op &op ) const { ///< op must return true to continue to read
        if ( off < end ) {
            // all the data is in the first buff
            if ( end <= beg->used )
                return op( beg, off, end );

            // use first buff
            const Buffer *b = beg, *n = b->next; // `n` and `e` must be computed now
            PT e = end - b->used;                // because op() may delete `beg`
            if ( not op( b, off, b->used ) )
                return false;

            if ( n ) {
                // following ones, in a 2x unrolled loop
                while ( true ) {
                    if ( e <= n->used )
                        return op( n, 0, e );
                    e -= n->used;
                    b = n->next;
                    if ( not op( n, 0, n->used ) )
                        return false;

                    if ( e <= b->used )
                        return op( b, 0, e );
                    e -= b->used;
                    n = b->next;
                    if ( not op( b, 0, b->used ) )
                        return false;
                }
            }
        }
        return true;
    }

    static PI8 read_byte_wo_dec_ref( const Buffer *&beg, PT &off, PT &end ) {
        PI8 res = beg->data[ off++ ];
        if ( off == beg->used ) {
            end -= beg->used; // != 0 because we know that end > off (previous test)
            beg = beg->next;
            off = 0;
        }
        return res;
    }

protected:
    friend class CbString;

    const Buffer *beg;
    PT            off; ///< beg marker
    PT            end; ///< end marker
};

} // namespace Hpipe
