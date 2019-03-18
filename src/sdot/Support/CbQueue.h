#pragma once

#include "Buffer.h"
#include "Assert.h"
#include "DaSi.h"

#include <functional>
#include <string>

namespace Hpipe {
class CbStringPtr;
class CbString;
class CmString;

/**
  Buffer sequence. Write to the end. Read from the beginning.
*/
class CbQueue {
public:
    CbQueue( CbQueue &&cq );
    CbQueue();
    ~CbQueue();

    CbQueue( const CmString &bs );
    CbQueue( const DaSi     &bs );

    CbQueue &operator=( CbQueue &&cq );

    // error
    // operator bool() const { return not error(); }
    bool error() const { return not end; } ///< works after at least a first read (and before free or clear): if no error, `end` is not cleared
    bool ack_error() { free(); end = 0; return false; } ///< set error flag to true, and return false

    // size
    bool empty() const { return not beg; }
    PT   size () const { PT res = 0; for( Buffer *b = beg; b; b = b->next ) res += b->used; return res - off; }
    PT   nbuf () const { PT res = 0; for( Buffer *b = beg; b; b = b->next ) ++res; return res; }
    void clear(); ///< set size( 0 ), and tries to keep the first Buffer if not shared
    void free ();  ///< set size( 0 ), and free all the data

    // writers
    void write_some( const void *data, PT size ); ///< append raw binary data
    void write_some( const CbStringPtr &s );
    void write_some( const CbString &s );
    void write_some( const CbQueue &s );
    void write_some( const std::string &s );
    void write_some( CbQueue &&cq );

    void write_byte( PI8 val ) {
        if ( not beg ) init_beg(); else if ( not end->room() ) end = Buffer::New( Buffer::default_size, end );
        end->data[ end->used++ ] = val;
    }

    void write_byte_wo_beg_test( PI8 val ) { ///< can be used after a first write( len != 0 ) or a first write_byte
        if ( not end->room() ) end = Buffer::New( Buffer::default_size, end );
        end->data[ end->used++ ] = val;
    }

    // void add_buff( IKnowWhatIDo, Buffer *buff, unsigned offset_in_buff = 0 ); ///< after that, buff should not be reused elsewhere

    void insert_some( PT pos, const void *data, PT size );

    // contiguous writers
    template<class T>
    T *write_cont( const T &data ) { ///< write contiguous
        return reinterpret_cast<T *>( write_cont( &data, sizeof( T ) ) );
    }

    void *write_cont( const void *data, PT size ); ///< write contiguous

    void *make_room( PT size ); ///< make contiguous room

    void sub_used( PT size ); ///< after make contiguous room, we can say how much data have not been used

    void *ptr( PT offset );
    const PI8 *ptr() const { return beg->data + off; }

    void new_buff( PT size = Buffer::default_size ) { if ( not beg ) init_beg(); else end = Buffer::New( size, end ); } ///< should not be called manually (expected for test purpose)
    CbQueue splitted( PT n ) const; ///< for test purpose. Make a newCbQueue with a copy of the content, splitted by chunks of size n

    // readers. Beware there are no checks in these methods
    void read_some( void *data, PT size );
    void skip_some( PT size );
    void skip_some_sr( ssize_t &size );

    PI8 read_byte() {
        PI8 res = beg->data[ off++ ];
        if ( off == beg->used ) {
            const Buffer *o = beg;
            beg = beg->next;
            Buffer::dec_ref( o );
            off = 0;
        }
        return res;
    }

    // checkings for readers that save a signal (that will give error() != 0) if not ok. To be done before each read.
    bool ack_read_byte() { ///< return true if ok to read a byte. Else, set end to 0 (to signal an error) and return false.
        if ( not beg )
            return ack_error();
        return true;
    }

    bool ack_read_some( ST len ) { ///< return true if ok to `len` bytes. Else, set end to 0 (to signal an error) and return false.
        len += off;
        for( Buffer *b = beg; b; b = b->next ) {
            len -= b->used;
            if ( len <= 0 )
                return true;
        }
        return ack_error();
    }

    //
    std::pair<const void *,PT> contiguous_data_at_the_beginning() {
        return beg ? std::pair<const void *,PT>( beg->data + off, beg->used - off ) : std::pair<const void *,PT>( 0, 0 );
    }

    //
    void data_visitor( std::function<void(const PI8 *,const PI8 *)> func ) const { ///< op must return true to continue to read
        visitor( [ &func ]( const Buffer *b, PT bb, PT eb ) {
            func( b->data + bb, b->data + eb );
            return true;
        } );
    }

    // display
    void write_to_stream( std::ostream &os ) const;

    operator std::string() const;

    //
    template<class TV0,class TV1>
    void remove_chunks( const TV0 &positions, const TV1 &sizes ) {
        Buffer  *buf_writer = beg;
        unsigned off_writer = off;
        PT       acc_pos = 0;
        PT       cur_pos = 0;
        PT       avoid   = 0;
        visitor( [ & ]( const Buffer *b, PT bd, PT ed ) { // buf_writer, &off_writer, &acc_pos, &cur_pos, &avoid, &positions, &sizes
            // TODO: optimize
            for( ; bd < ed; ++bd, ++acc_pos ) {
                if ( cur_pos < positions.size() and acc_pos == positions[ cur_pos ] )
                    avoid = sizes[ cur_pos++ ];

                if ( not avoid ) {
                    buf_writer->data[ off_writer++ ] = b->data[ bd ];
                    if ( off_writer >= buf_writer->used ) {
                        buf_writer = buf_writer->next;
                        off_writer = 0;
                    }
                } else
                    --avoid;
            }
            return true;
        } );

        // dec_ref unused buffers
        for( const Buffer *n = buf_writer->next; n; ) {
            const Buffer *o = n;
            n = n->next;
            Buffer::dec_ref( o );
        }

        // update the last one
        buf_writer->used = off_writer;
        buf_writer->next = 0;
    }


protected:
    friend class CbStringPtr;
    friend class CbString;

    void init_beg() {
        beg = Buffer::New();
        end = beg;
        off = 0;
    }

    template<class Op>
    bool visitor( const Op &op ) const { ///< op must return true to continue to read
        if ( const Buffer *b = beg ) {
            // first buff
            const Buffer *n = b->next;
            if ( not op( b, off, b->used ) )
                return false;

            // next ones, 2x unrolled
            while ( n ) {
                b = n->next;
                if ( not op( n, 0, n->used ) )
                    return false;
                if ( not b )
                    break;

                n = b->next;
                if ( not op( b, 0, b->used ) )
                    return false;
            }
        }
        return true;
    }

public:
    Buffer  *beg; ///< first item
    Buffer  *end; ///< last item
    unsigned off; ///< offset in bytes (< Buffer::size)
};

}
