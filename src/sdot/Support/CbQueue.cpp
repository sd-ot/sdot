#include "CbStringPtr.h"
#include "CmString.h"
#include "CbQueue.h"
#include "Assert.h"
#include <string.h>

using namespace Hpipe;

CbQueue::CbQueue( CbQueue &&cq ) : beg( cq.beg ), end( cq.end ), off( cq.off ) {
    new ( &cq ) CbQueue;
}

CbQueue::CbQueue() : beg( 0 ), end( (Buffer *)8 /*because error() have to return false*/ ), off( 0 ) {
}

CbQueue::~CbQueue() {
    for( const Buffer *b = beg; b; ) {
        const Buffer *o = b;
        b = b->next;
        Buffer::dec_ref( o );
    }
}

CbQueue::CbQueue( const CmString &bs ) : CbQueue() {
    write_some( bs.ptr(), bs.size() );
}

CbQueue::CbQueue( const DaSi &bs ) : CbQueue() {
    write_some( bs.data, bs.size );
}

CbQueue &CbQueue::operator=( CbQueue &&cq ) {
    this->~CbQueue();
    new ( this ) CbQueue( std::move( cq ) );
    return *this;
}

void CbQueue::clear() {
    // try to find a buffer that can be reused. dec_ref the others
    for( Buffer *b = beg; b; b = b->next ) {
        if ( b->cpt_use == 0 ) {
            Buffer *n = b->next;
            b->used = 0;
            b->next = 0;
            beg = b;
            end = b;
            off = 0;
            for( Buffer *o; n; n = o ) {
                o = n->next;
                Buffer::dec_ref( n );
            }
            return;
        }
        // dec_ref is not going to delete `b`, meaning that `b->next` is ok after that
        Buffer::dec_ref( b );
    }

    // init (because we haven't found any reusable buffer)
    beg = 0;
    off = 0;
}

void CbQueue::free() {
    // dec_ref all the buffers
    for( const Buffer *b = beg, *n; b; b = n ) {
        n = b->next;
        Buffer::dec_ref( b );
    }

    // re-init
    beg = 0;
    off = 0;
}

CbQueue::operator std::string() const {
    std::string res; res.reserve( size() );

    visitor( [ &res ]( const Buffer *b, PT bl, PT el ) {
        res.append( b->data + bl, b->data + el );
        return true;
    } );

    return res;
}

void CbQueue::write_some( const void *data, PT size ) {
    if ( ! size )
        return;

    if ( beg ) {
        PT room = end->room();
        if ( size <= room ) {
            memcpy( end->data + end->used, data, size );
            end->used += size;
            return;
        }
        // -> size > room, buffer with used that may be != 0
        memcpy( end->data + end->used, data, room ); data = reinterpret_cast<const PI8 *>( data ) + room;
        end->used = end->size;
        size -= room;

        end = Buffer::New( std::max( PT( Buffer::default_size ), size ), end );
    } else {
        beg = Buffer::New( std::max( PT( Buffer::default_size ), size ) );
        end = beg;
        off = 0;
    }

    // -> size != 0, buffers with used == 0
    while ( size > end->size ) {
        memcpy( end->data, data, end->size ); data = reinterpret_cast<const PI8 *>( data ) + end->size;
        end->used = end->size;
        size -= end->size;

        end = Buffer::New( Buffer::default_size, end );
    }

    // -> buffer with used == 0, size < Buffer::size
    memcpy( end->data, data, size );
    end->used = size;
}

void CbQueue::write_some( const CbStringPtr &s ) {
    s.data_visitor( [ this ]( const PI8 *b, const PI8 *e ) {
        write_some( b, e - b );
    } );
}

void CbQueue::write_some( const CbString &s ) {
    s.data_visitor( [ this ]( const PI8 *b, const PI8 *e ) {
        write_some( b, e - b );
    } );
}

void CbQueue::write_some( const CbQueue &s ) {
    s.data_visitor( [ this ]( const PI8 *b, const PI8 *e ) {
        write_some( b, e - b );
    } );
}

void CbQueue::write_some( const std::string &s ) {
    write_some( s.data(), s.size() );
}

void CbQueue::write_some( CbQueue &&cq ) {
    if ( cq.off ) {
        write_some( cq.beg->data + cq.off, cq.beg->used - cq.off );
        Buffer *n = cq.beg->next;
        Buffer::dec_ref( cq.beg );
        cq.beg = n;
    }

    if ( ! beg ) {
        beg = cq.beg;
        off = 0;
    } else {
        end->next = cq.beg;
    }
    end = cq.end;

    // we don't want to dec_ref the buffers of cq (which are now used in this)
    new ( &cq ) CbQueue;
}

//void CbQueue::add_buff( IKnowWhatIDo, Buffer *buff, unsigned offset_in_buff ) {
//    write_some( buff->data + offset_in_buff, buff->used - offset_in_buff ); // TODO: optimize
//    //    ++buff->cpt_use;
//    //    if ( not beg ) {
//    //        beg = buff;
//    //        end = buff;
//    //        off = 0;
//    //    } else {
//    //        end->next = buff;
//    //        end = buff;
//    //    }
//}

void CbQueue::insert_some( PT pos, const void *data, PT size ) {
    if ( off || pos ) {
        TODO;
    }
    // simple case: we want to insert stuff at the beginning of the first block
    Buffer *b = Buffer::New( size );
    memcpy( b->data, data, size );
    b->used = size;

    if ( ! beg ) end = b;
    b->next = beg;
    beg = b;
}

void *CbQueue::write_cont( const void *data, PT size ) {
    void *res = make_room( size );
    memcpy( res, data, size );
    return res;
}

void *CbQueue::make_room( PT size ) {
    if ( ! beg ) {
        beg = Buffer::New( std::max( PT( Buffer::default_size ), size ) );
        end = beg;
        off = 0;
    } else if ( size > end->room() )
        end = Buffer::New( std::max( PT( Buffer::default_size ), size ), end );

    void *res = end->data + end->used;
    end->used += size;
    return res;
}

void CbQueue::sub_used( PT size ) {
    beg->used -= size;
}

void *CbQueue::ptr( PT offset ) {
    if ( ! beg )
        return 0;

    // beg buffer
    if ( off + offset < beg->used )
        return beg->data + off + offset;
    offset -= beg->used - off;

    // next buffers
    for( Buffer *buf = beg->next; buf; buf = buf->next ) {
        if ( offset < buf->used )
            return buf->data + offset;
        offset -= buf->used;
    }
    return 0;
}

CbQueue CbQueue::splitted( PT n ) const {
    CbQueue res;
    std::string data = *this;
    for( PT i = 0; i < data.size(); i += n ) {
        res.new_buff( n );
        res.write_some( &data[ 0 ] + i, std::min( i + n, data.size() ) - i );
    }
    return res;
}

void CbQueue::read_some( void *data, PT size ) {
    visitor( [ this, &data, &size ]( const Buffer *b, PT bd, PT ed ) {
        PT lm = ed - bd;
        if ( size < lm ) {
            memcpy( data, b->data + bd, size );
            off += size;
            return false;
        }

        size -= lm;
        memcpy( data, b->data + off, lm );
        data = reinterpret_cast<PI8 *>( data ) + lm;

        beg = b->next;
        off = 0;

        Buffer::dec_ref( b );
        return true;
    } );
}

void CbQueue::skip_some( PT size ) {
    visitor( [ this, &size ]( const Buffer *b, PT bd, PT ed ) {
        PT lm = ed - bd;
        if ( size < lm ) {
            off += size;
            return false;
        }

        beg = b->next;
        Buffer::dec_ref( b );
        size -= lm;
        off = 0;
        return true;
    } );
}

void CbQueue::skip_some_sr( std::ptrdiff_t &size ) {
    if ( size <= 0 )
        return;
    visitor( [ this, &size ]( const Buffer *b, PT bd, PT ed ) {
        PT lm = ed - bd;
        if ( PT( size ) < lm ) {
            off += size;
            size = 0;
            return false;
        }

        beg = b->next;
        Buffer::dec_ref( b );
        size -= lm;
        off = 0;
        return true;
    } );
}

void CbQueue::write_to_stream( std::ostream &os ) const {
    int cpt = 0;
    visitor( [ & ]( const Buffer *b, PT beg, PT end ) {
        static const char *c = "0123456789abcdef";
        for( PT i = beg; i < end; ++i )
            os << ( cpt++ ? " " : "" ) << c[ b->data[ i ] / 16 ] << c[ b->data[ i ] % 16 ];
        return true;
    } );
}


#ifdef Hpipe_JS
#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#include "BinStream.h"
using namespace emscripten;
using namespace Hpipe;

EMSCRIPTEN_BINDINGS( CbQueue_module ) {
    class_<CbQueue>( "CbQueue" )
            .constructor<>()
            //.property( "error"     , &CbQueue::error )
            //.property( "empty"     , &CbQueue::empty )
            //.property( "size"      , &CbQueue::size )

            .function( "write_some", select_overload<void(const std::string &s)>( &CbQueue::write_some ) )

            .function( "to_string" , &CbQueue::operator std::string )
            //.class_function("getStringFromInstance", &MyClass::getStringFromInstance)
            ;

    class_<BinStream<CbQueue>>( "BinStream_CbQueue" )
            .constructor<CbQueue *>()
            //.property( "error"         , &BinStream<CbQueue>::error )
            //.property( "empty"         , &BinStream<CbQueue>::empty )
            // .property( "size"          , &BinStream<CbQueue>::size  )

            .function( "write_unsigned", &BinStream<CbQueue>::write_unsigned<int> )
            .function( "write_signed"  , &BinStream<CbQueue>::write_signed  <int> )
            .function( "write_byte"    , &BinStream<CbQueue>::write_byte          )

            .function( "read_byte"     , &BinStream<CbQueue>::read_byte )
            ;
}
#endif
