#pragma once

#include "CmString.h"
#include "CbQueue.h"

namespace Hpipe {
class CbStringPtr;

/**
  Part of a buffer sequence. Read-only.

  Can be used as an input stream.

  Assumptions: we retain (ref count) a buffer only if we use data from it
*/
class CbString {
public:
    struct NoIncRef {};

    CbString( const CbString    &bs, PT off, PT len );
    CbString( const CbString    &bs );

    CbString( const CbStringPtr &bs, PT off, PT len );
    CbString( const CbStringPtr &bs );

    CbString( CbString         &&bs, PT off, PT len );
    CbString( CbString         &&bs ) : beg( bs.beg ), off( bs.off ), end( bs.end ) { bs.off = 0; bs.end = 0; }

    CbString( const CbQueue     &bs, PT s_off, PT s_len );
    CbString( const CbQueue     &bs );

    CbString( CbQueue          &&bs, PT off, PT len );
    CbString( CbQueue          &&bs ) : beg( bs.beg ), off( bs.off ), end( off + bs.size() ) { bs.beg = 0; bs.end = 0; bs.off = 0; }

    CbString( const CmString    &bs ) : CbString( CbQueue( bs ) ) {}
    CbString( const DaSi        &ds ) : CbString( CbQueue( ds ) ) {}

    CbString( const std::string &bs );

    CbString( NoIncRef, const Buffer *beg_buff, const Buffer::PI8 *beg_data, const Buffer *end_buff, const Buffer::PI8 *end_data ); ///< this constructor does not inc_ref the buffers
    CbString( Buffer *buff, PT off, PT len ); ///< this constructor does not inc_ref the buffers

    void inc_length_wo_cr() { ++end; } ///< dangerous (ref count is managerd elsewhere)

    CbString() : beg( 0 ), off( 0 ), end( 0 ) {}

    ~CbString() { dec_ref_buffs(); }

    void operator=( CbString &&cs ) { dec_ref_buffs(); beg = cs.beg; off = cs.off; end = cs.end; cs.off = cs.end; }

    explicit operator bool() const { return not empty(); }

    void free();

    const Buffer *get_buf() { return beg; }
    PT            get_off() { return off; }
    PT            get_end() { return end; }

    // errors
    // operator bool() const { return not error(); }
    bool error() const { return ST( off ) < 0; }
    bool ack_error();

    // size
    bool empty() const { return end == off; }
    PT   nbuf () const { PT res = 0; for( const Buffer *b = beg; b; b = b->next ) ++res; return res; }
    PT   size () const { return end - off; }

    CbString substring( PT off, PT len ) const { return { *this, off, len }; }
    CbString beg_upto( PT len ) const { return substring( 0, len ); }
    CbString end_from( PT off ) const { PT s = size(); off = std::min( off, s ); return substring( off, s - off ); }

    // readers. Beware there are no checks in these methods (ack_read_... must be called before, to check if possible)
    void read_some( void *data, PT size );

    PI8 read_byte() {
        PI8 res = beg->data[ off++ ];
        if ( off == end ) {
            Buffer::dec_ref( beg );
        } else if ( off == beg->used ) {
            const Buffer *o = beg;
            end -= beg->used; // != 0 because we know that end > off (previous test)
            beg = beg->next;
            off = 0;
            Buffer::dec_ref( o );
        }
        return res;
    }

    CbString read_line( char sep = '\n', bool skip_void_lines = true );

    CbString &skip_some   ( PT size );
    CbString &skip_some_sr( ssize_t &size );
    CbString &skip_byte   () { read_byte(); return *this; }

    const PI8 *ptr() const { return beg->data + off; }



    // checkings for readers that save a signal (that will give error() != 0) if not ok. To be done before each read.
    bool ack_read_byte()         { if ( off   >=    end ) return ack_error(); return true; } ///< return true if ok to read a byte. Else, set end to 0 (to signal an error) and return false.
    bool ack_read_some( PT len ) { if ( off + len > end ) return ack_error(); return true; } ///< return true if ok to `len` bytes. Else, set end to 0 (to signal an error) and return false.

    // display
    operator std::string() const;

    //
    void data_visitor( std::function<void(const PI8 *,const PI8 *)> func ) const { ///< op must return true to continue to read
        visitor( [ &func ]( const Buffer *b, ST bb, ST eb ) {
            func( b->data + bb, b->data + eb );
            return true;
        } );
    }

    bool data_visitor( std::function<bool(const PI8 *,const PI8 *)> func ) const { ///< op must return true to continue to read
        return visitor( [ &func ]( const Buffer *b, ST bb, ST eb ) {
            return func( b->data + bb, b->data + eb );
        } );
    }

    bool starts_with( const char *data );
    bool starts_with( const void *data, PT size );

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
        int cpt = 0;
        visitor( [ &os, &cpt ]( const Buffer *b, PT beg, PT end ) {
            os.write( (const char *)b->data + beg, end - beg );
            //for( PT i = beg; i < end; ++i )
            //    os << ( cpt++ ? " " : "" ) << unsigned( b->data[ i ] );
            //        static const char *c = "0123456789abcdef";
            //        for( PT i = beg; i < end; ++i )
            //            os << ( cpt++ ? " " : "" ) << c[ b->data[ i ] / 16 ] << c[ b->data[ i ] % 16 ];
            return true;
        } );
    }

protected:
    friend class CbStringPtr;

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

    void dec_ref_buffs() {
        // nothing to dec_ref ?
        if ( off == end )
            return;
        // else, dec_ref until `end`
        const Buffer *b = beg;
        PT e = end;
        while( b ) {
            unsigned used = b->used;
            const Buffer *o = b;
            b = b->next;
            Buffer::dec_ref( o );
            if ( e <= used )
                break;
            e -= used;
        }
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

    const Buffer *beg;
    PT            off; ///< beg marker
    PT            end; ///< end marker
};

} // namespace Hpipe
