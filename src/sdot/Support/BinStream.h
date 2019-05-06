#pragma once

#include "CbStringPtr.h"
#include "CbString.h"
#include "CmString.h"
#include "CbQueue.h"
#include "CmQueue.h"
#include "CfQueue.h"
#include <string.h>
#include <vector>
#include <array>

namespace Hpipe {

/**
  To read || write from/to CmQueue, CbString, CmString, ...
*/
template<class TB=CbQueue>
struct BinStream {
    BinStream( TB *buf ) : buf( buf ) {}

    operator bool() const { return ! buf->error(); }
    bool error() const { return buf->error(); }
    bool empty() const { return buf->empty(); }
    ST size() const { return buf->size(); }

    // unsigned int writers
    BinStream &write_unsigned( PI8 val ) {
        return write_byte( val );
    }

    template<class T>
    BinStream &write_unsigned( T val ) {
        // ASSERT_IF_DEBUG( val >= 0 );
        if ( val < 128 ) {
            buf->write_byte( val );
            return *this;
        }

        buf->write_byte( 128 + val % 128 );
        val >>= 7;

        for( ; val >= 128; val >>= 7 )
            buf->write_byte_wo_beg_test( 128 + val % 128 );
        buf->write_byte_wo_beg_test( val );
        return *this;
    }

    template<class T,std::size_t si>
    BinStream &write_unsigned( const std::array<T,si> &val ) {
        for( const T &v : val )
            write_unsigned( v );
        return *this;
    }

    template<class T>
    BinStream &write_unsigned( const std::vector<T> &val ) {
        write_unsigned( val.size() );
        for( const T &v : val )
            write_unsigned( v );
        return *this;
    }

    // signed int writers
    BinStream &write_signed( SI8 val ) {
        buf->write_byte( val );
        return *this;
    }

    template<class T>
    BinStream &write_signed( T val ) {
        int a = 0;
        if ( val < 0 ) {
            val = - val;
            a = 128;
        }

        if ( val < 64 ) {
            buf->write_byte( a + val );
            return *this;
        }

        buf->write_byte( a + 64 + val % 64 );
        val /= 64;

        for( ; val >= 128; val /= 128 )
            buf->write_byte_wo_beg_test( 128 + val % 128 );
        buf->write_byte_wo_beg_test( val );
        return *this;
    }

    template<class T,std::size_t si>
    BinStream &write_signed( const std::array<T,si> &val ) {
        for( const T &v : val )
            write_signed( v );
        return *this;
    }

    template<class T>
    BinStream &write_signed( const std::vector<T> &val ) {
        write_unsigned( val.size() );
        for( const T &v : val )
            write_signed( v );
        return *this;
    }

    // generic writers
    BinStream &write_some( const void *ptr, ST len ) {
        buf->write_some( ptr, len );
        return *this;
    }

    BinStream &write_some( const char *ptr ) {
        return write_some( ptr, strlen( ptr ) );
    }

    BinStream &write_some( const CbString &s ) {
        buf->write_some( s );
        return *this;
    }

    BinStream &write_some( const CmString &s ) {
        buf->write_some( s.ptr(), s.size() );
        return *this;
    }

    BinStream &write_some( const CbQueue &s ) {
        buf->write_some( s );
        return *this;
    }

    BinStream &write_some( CbQueue &&s ) {
        buf->write_some( std::move( s ) );
        return *this;
    }

    BinStream &write_byte( PI8 val ) {
        buf->write_byte( val );
        return *this;
    }

    BinStream &write_String( const void *val, size_t len ) {
        return write_unsigned( len ).write_some( val, len );
    }

    template<class T>
    BinStream &write_le( T val ) { // little endian
        for( PT i = 0; i < sizeof( val ); ++i, val >>= 8 )
            buf->write_byte( val );
        return *this;
    }

    template<class T>
    BinStream &write_be( T val ) { // big endian (network order)
        for( PT i = sizeof( val ); i -= 8; )
            buf->write_byte( val >> i );
        return *this;
    }

    BinStream &write_be16( PI16 val ) { // big endian (network order)
        buf->write_byte( val >> 8 );
        buf->write_byte( val >> 0 );
        return *this;
    }

    void flush() {
        buf->flush();
    }

    // size needed for unsigned_...
    static ST size_needed_for_unsigned( PI8 val ) {
        return 1;
    }

    template<class T>
    static ST size_needed_for_unsigned( T val ) {
        // ASSERT_IF_DEBUG( val >= 0 );
        ST res = 1;
        for( ; val >= 128; val /= 128 )
            ++res;
        return res;
    }

    // size needed for signed_...
    static ST size_needed_for_signed( SI8 val ) {
        return 1;
    }

    template<class T>
    static ST size_needed_for_signed( T val ) {
        TODO;
        return 0;
    }

    // helpers for size needed for ...
    static ST size_needed_for( PI8  val ) { return 1; }
    static ST size_needed_for( PI16 val ) { return size_needed_for_unsigned( val ); }
    static ST size_needed_for( PI32 val ) { return size_needed_for_unsigned( val ); }
    static ST size_needed_for( PI64 val ) { return size_needed_for_unsigned( val ); }

    static ST size_needed_for( SI8  val ) { return 1; }
    static ST size_needed_for( SI16 val ) { return size_needed_for_signed( val ); }
    static ST size_needed_for( SI32 val ) { return size_needed_for_signed( val ); }
    static ST size_needed_for( SI64 val ) { return size_needed_for_signed( val ); }

    // simplified writers
    template<class T>
    BinStream &operator<<( const T              &s     ) { s.write_to( *this ); return *this; }

    BinStream &operator<<( const Bool           &val   ) { return write_byte( val ); }

    BinStream &operator<<( const PI8            &val   ) { return write_unsigned( val ); }
    BinStream &operator<<( const PI16           &val   ) { return write_unsigned( val ); }
    BinStream &operator<<( const PI32           &val   ) { return write_unsigned( val ); }
    BinStream &operator<<( const unsigned long  &val   ) { return write_unsigned( val ); } /// hum

    BinStream &operator<<( const SI8            &val   ) { return write_signed( val ); }
    BinStream &operator<<( const SI16           &val   ) { return write_signed( val ); }
    BinStream &operator<<( const SI32           &val   ) { return write_signed( val ); }
    BinStream &operator<<( const SI64           &val   ) { return write_signed( val ); }

    BinStream &operator<<( const FP32           &val   ) { return write_some( &val, sizeof( FP32 ) ); }
    BinStream &operator<<( const FP64           &val   ) { return write_some( &val, sizeof( FP64 ) ); }

    BinStream &operator<<( const char           *c_str ) { ST s = strlen( c_str ); write_unsigned( s ); return write_some( c_str, s ); }
    BinStream &operator<<( const std::string    &str   ) { write_unsigned( str.size() ); return write_some( str.data(), str.size() ); }
    BinStream &operator<<( const CbStringPtr    &s     ) { write_unsigned( s.size() ); return write_some( s ); }
    BinStream &operator<<( const CbString       &s     ) { write_unsigned( s.size() ); return write_some( s ); }
    BinStream &operator<<( const CmString       &s     ) { write_unsigned( s.size() ); return write_some( s ); }
    BinStream &operator<<( const CbQueue        &s     ) { write_unsigned( s.size() ); return write_some( s ); }
    BinStream &operator<<( CbQueue             &&s     ) { write_unsigned( s.size() ); return write_some( std::move( s ) ); }

    template<class T>
    BinStream &operator<<( const std::vector<T> &s     ) { write_unsigned( s.size() ); for( const T &v : s ) operator<<( v ); return *this; }

    template<class T,std::size_t size>
    BinStream &operator<<( const std::array<T,size> &s ) { for( const T &v : s ) operator<<( v ); return *this; }

    // skip (data we don't want to read)
    BinStream &skip_some( PT len ) {
        buf->skip_some( len );
        return *this;
    }

    BinStream &skip_unsigned() {
        while ( buf->ack_read_byte() and buf->read_byte() >= 128 ) { }
        return *this;
    }

    BinStream &skip_string() {
        return skip_some( read_unsigned() );
    }

    //    template<class T>
    //    BinStream &skip( TypeList<T       > ) { read().operator typename Unref<T>::T(); return *this; }

    //    BinStream &skip( TypeList<PI8     > ) { return skip_some( 1 ); }
    //    BinStream &skip( TypeList<PI16    > ) { return skip_unsigned(); }
    //    BinStream &skip( TypeList<PI32    > ) { return skip_unsigned(); }
    //    BinStream &skip( TypeList<PI64    > ) { return skip_unsigned(); }
    //    BinStream &skip( TypeList<CbString> ) { return skip_string(); }

    // generic readers
    void read_some( void *ptr, ST len ) {
        if ( ! buf->ack_read_some( len ) )
            return;
        return buf->read_some( ptr, len );
    }

    PI8 read_byte() {
        if ( ! buf->ack_read_byte() )
            return 0;
        return buf->read_byte();
    }

    template<class T,std::size_t si>
    BinStream &read_unsigned( std::array<T,si> &val ) {
        for( T &v : val )
            read_unsigned( v );
        return *this;
    }

    template<class T>
    BinStream &read_unsigned( std::vector<T> &val ) {
        val.resize( read_unsigned() );
        for( T &v : val )
            read_unsigned( v );
        return *this;
    }

    // unsigned int readers
    template<class T>
    void read_unsigned( T &res ) {
        if ( ! buf->ack_read_byte() )
            return;
        res = buf->read_byte();
        if ( res < 128 )
            return;
        // -> "big" number
        res -= 128;
        int shift = 7;
        while ( true ) {
            if ( ! buf->ack_read_byte() )
                return;
            T v = buf->read_byte();
            if ( v < 128 ) {
                res += ( v << shift );
                return;
            }
            res += ( v - 128 ) << shift;
            shift += 7;
        }
    }

    void read_unsigned( PI8 &res ) {
        if ( ! buf->ack_read_byte() )
            return;
        res = buf->read_byte();
    }

    template<class T>
    void read_unsigned( T &res, ST &l ) { ///< --l each time a char is read
        if ( ! buf->ack_read_byte() )
            return;
        res = buf->read_byte(); --l;
        if ( res < 128 )
            return;
        // "big" number
        res -= 128;
        int shift = 7;
        while ( true ) {
            if ( ! buf->ack_read_byte() )
                return;
            T v = buf->read_byte(); --l;
            if ( v < 128 ) {
                res += ( v << shift );
                return;
            }
            res += ( v - 128 ) << shift;
            shift += 7;
        }
    }

    // signed int readers
    template<class T>
    void read_signed( T &res ) {
        if ( ! buf->ack_read_byte() )
            return;
        res = buf->read_byte();

        // negative ?
        if ( res >= 128 ) {
            res -= 128;
            if ( res < 64 ) {
                res = -res;
                return;
            }
            res = 64 - res;
            int shift = 6;
            while ( true ) {
                if ( ! buf->ack_read_byte() )
                    return;
                T v = buf->read_byte();
                if ( v < 128 ) {
                    res -= v << shift;
                    return;
                }
                res -= ( v - 128 ) << shift;
                shift += 7;
            }
        }

        // -> positive
        if ( res < 64 )
            return;
        res -= 64;
        int shift = 6;
        while ( true ) {
            if ( ! buf->ack_read_byte() )
                return;
            T v = buf->read_byte();
            if ( v < 128 ) {
                res += ( v << shift );
                return;
            }
            res += ( v - 128 ) << shift;
            shift += 7;
        }
    }

    template<class T>
    void read_signed( T &res, ST &l ) {
        if ( ! buf->ack_read_byte() )
            return;
        res = buf->read_byte(); --l;

        // negative ?
        if ( res >= 128 ) {
            res -= 128;
            if ( res < 64 ) {
                res = -res;
                return;
            }
            res = 64 - res;
            int shift = 6;
            while ( true ) {
                if ( ! buf->ack_read_byte() )
                    return;
                T v = buf->read_byte(); --l;
                if ( v < 128 ) {
                    res -= v << shift;
                    return;
                }
                res -= ( v - 128 ) << shift;
                shift += 7;
            }
        }

        // -> positive
        if ( res < 64 )
            return;
        res -= 64;
        int shift = 6;
        while ( true ) {
            if ( ! buf->ack_read_byte() )
                return;
            T v = buf->read_byte();  --l;
            if ( v < 128 ) {
                res += ( v << shift );
                return;
            }
            res += ( v - 128 ) << shift;
            shift += 7;
        }
    }

    template<class T,std::size_t si>
    BinStream &read_signed( std::array<T,si> &val ) {
        for( T &v : val )
            read_signed( v );
        return *this;
    }

    template<class T>
    BinStream &read_signed( std::vector<T> &val ) {
        val.resize( read_unsigned() );
        for( T &v : val )
            read_signed( v );
        return *this;
    }

    template<class T>
    void read_le( T &res ) {
        res = 0;
        for( PT i = 0, s = 0; i < sizeof( T ); ++i, s += 8 )
            res |= T( read_byte() ) << s;
    }

    PI16 read_be16() {
        return ( read_byte() << 8 ) | read_byte();
    }

    PI32 read_be32() {
        return ( read_byte() << 24 ) | ( read_byte() << 16 ) | ( read_byte() << 8 ) | read_byte();
    }

    //
    CbString read_CbString( ST max_size ) {
        PI64 size = read();
        if ( error() )
            return {};
        if ( size > max_size ) {
            buf->ack_error();
            return {};
        }
        if ( ! buf->ack_read_some( size ) )
            return CbString();
        CbString res( *buf, 0, size );
        skip_some( size );
        return res;
    }

    CbString read_CbString() {
        PI64 size = read();
        if ( error() || ! buf->ack_read_some( size ) )
            return CbString();
        CbString res( *buf, 0, size );
        skip_some( size );
        return res;
    }

    std::string read_String( ST max_size ) {
        PI64 size = read();
        if ( error() )
            return std::string();
        if ( size > max_size ) {
            buf->ack_error();
            return {};
        }
        if ( ! buf->ack_read_some( size ) )
            return {};
        std::string res;
        res.resize( size );
        read_some( &res[ 0 ], size );
        return res;
    }

    std::string read_String() {
        PI64 size = read();
        if ( error() || ! buf->ack_read_some( size ) )
            return {};
        std::string res;
        res.resize( size );
        read_some( &res[ 0 ], size );
        return res;
    }

    //    DaSi read_DaSi() { ///< works only for CmQueue, CmString || similar (assumes that the data won't be freed)
    //        DaSi res;
    //        res.size = read();
    //        if ( error() || ! buf->ack_read_some( res.size ) )
    //            return { nullptr, 0 };
    //        res.data = (const char *)buf->ptr();
    //        skip_some( res.size );
    //        return res;
    //    }

    CmString read_CmString() { ///< works only for CmQueue, CmString || similar (assumes that the data won't be freed)
        PT size = read();
        if ( error() || ! buf->ack_read_some( size ) )
            return { (const void *)0, (const void *)0 };
        const PI8 *data = buf->ptr();
        skip_some( size );
        return { data, data + size };
    }

    //
    const PI8 *ptr() const { return buf->ptr(); }

    // std notations
    template<class T>
    BinStream &operator>>( T &val ) {
        val = read();
        return *this;
    }

    // readers like T res = b.read() (or b.read_unsigned(), ...)
    struct UnsignedWulReader {
        UnsignedWulReader( BinStream *b, ST &len ) : b( b ), len( len ) {}
        template<class T> operator T() { T res = 0; b->read_unsigned( res, len ); return res; }
        BinStream *b;
        ST &len;
    };
    struct SignedWulReader {
        SignedWulReader( BinStream *b, ST &len ) : b( b ), len( len ) {}
        template<class T> operator T() { T res = 0; b->read_signed( res, len ); return res; }
        BinStream *b;
        ST &len;
    };
    struct UnsignedReader {
        UnsignedReader( BinStream *b ) : b( b ) {}
        template<class T,std::size_t s> operator std::array<T,s>() { std::array<T,s> res{}; for( T &val : res ) b->read_unsigned( val ); return res; }
        template<class T> operator std::vector<T>() { std::size_t size = b->read_unsigned(); std::vector<T> res( size ); for( T &val : res ) b->read_unsigned( val ); return res; }
        template<class T> operator T() { T res = 0; b->read_unsigned( res ); return res; }
        BinStream *b;
    };
    struct SignedReader {
        SignedReader( BinStream *b ) : b( b ) {}
        template<class T,std::size_t s> operator std::array<T,s>() { std::array<T,s> res; for( T &val : res ) b->read_signed( val ); return res; }
        template<class T> operator T() { T res = 0; b->read_signed( res ); return res; }
        BinStream *b;
    };
    struct LeReader {
        LeReader( BinStream *b ) : b( b ) {}
        template<class T> operator T() { T res = 0; b->read_le( res ); return res; }
        BinStream *b;
    };
    struct GenReader {
        GenReader( BinStream *b ) : b( b ) {}

        template<class T>
        operator T             () { return T::read_from( *b ); }

        template<class T>
        operator std::vector<T>() { std::size_t size = b->read_unsigned(); std::vector<T> res( size ); for( T &val : res ) val = b->read(); return res; }

        template<class T,std::size_t s>
        operator std::array<T,s>() { std::array<T,s> res; for( T &val : res ) val = b->read(); return res; }

        operator CbString      () { return b->read_CbString(); }
        operator CmString      () { return b->read_CmString(); }
        operator std::string   () { return b->read_String  (); }
        // operator DaSi          () { return b->read_DaSi    (); }

        operator Bool          () { return b->read_byte(); }

        operator PI8           () { return b->read_unsigned(); }
        operator PI16          () { return b->read_unsigned(); }
        operator PI32          () { return b->read_unsigned(); }
        operator PI64          () { return b->read_unsigned(); }

        operator SI8           () { return b->read_signed(); }
        operator SI16          () { return b->read_signed(); }
        operator SI32          () { return b->read_signed(); }
        operator SI64          () { return b->read_signed(); }

        #ifdef __llvm__
        operator unsigned long () { return b->read_unsigned(); }
        #endif

        operator FP32          () { FP32 val = 0; b->read_some( &val, sizeof( val ) ); return val; }
        operator FP64          () { FP64 val = 0; b->read_some( &val, sizeof( val ) ); return val; }

        BinStream *b;
    };

    UnsignedWulReader read_unsigned_wul( ST &l ) { return UnsignedWulReader( this, l ); } ///< l -= read size
    SignedWulReader   read_signed_wul( ST &l ) { return SignedWulReader  ( this, l ); } ///< l -= read size
    UnsignedReader    read_unsigned() { return this; }
    SignedReader      read_signed() { return this; }
    LeReader          read_le() { return this; }
    GenReader         read() { return this; }

    TB *buf;
};

using BBQ = BinStream<CbQueue>;

} // namespace Hpipe
