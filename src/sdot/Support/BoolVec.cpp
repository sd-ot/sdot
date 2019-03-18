#include "BoolVec.h"
#include "Assert.h"
#include <string.h>

BoolVec::BoolVec( Reference, void *data, size_t size ) : _data( (W *)data ), _size( size ), _rese( 0 ) {
}

BoolVec::BoolVec( void *data, size_t size ) : BoolVec( size ) {
    memcpy( this->_data, data, ( size + 7 ) / 8 );
}

BoolVec::BoolVec( size_t size, bool val ) : BoolVec( size ) {
    if ( val )
        set_all();
    else
        clear_all();
}

BoolVec::BoolVec( const BoolVec &bv ) : BoolVec( bv._data, bv._size ) {
}

BoolVec::BoolVec( size_t size ) : _size( size ) {
    size_t bs = ( size + nbw - 1 ) / nbw;
    _rese = nbw * bs;
    if ( size )
        _data = (W *)malloc( sizeof( W ) * bs );
}

BoolVec::~BoolVec() {
    if ( _rese )
        free( _data );
}

BoolVec &BoolVec::operator=( const BoolVec &that ) {
    resize( that._size );
    memcpy( _data, that._data, ( that._size + 7 ) / 8 );
    return *this;
}

BoolVec &BoolVec::operator|=( const BoolVec &that ) {
    ASSERT( _size == that._size, "..." );
    for( size_t i = 0; i < ( _size + nbw - 1 ) / nbw; ++i )
        _data[ i ] |= that._data[ i ];
    return *this;
}

BoolVec &BoolVec::operator&=( const BoolVec &that ) {
    ASSERT( _size == that._size, "..." );
    for( size_t i = 0; i < ( _size + nbw - 1 ) / nbw; ++i )
        _data[ i ] &= that._data[ i ];
    return *this;
}

void BoolVec::write_to_stream( std::ostream &os ) const {
    for( size_t i = 0; i < _size; ++i )
        os << get( i );
}

void BoolVec::resize( size_t new_size ) {
    if ( _rese < new_size ) {
        size_t new_blen = ( new_size + nbw - 1 ) / nbw;
        size_t new_rese = nbw * new_blen;
        W     *new_data = (W *)malloc( sizeof( W ) * new_blen );
        if ( _size )
            memcpy( new_data, _data, ( _size + 7 ) / 8 );
        if ( _rese )
            free( _data );
        _data = new_data;
        _rese = new_rese;
        _size = new_size;
    }
}

//bool BoolVec::operator==( const BoolVec &that ) const {
//    return size == that.size && memcmp_bit( data, 0, that.data, 0, size ) == 0;
//}

//bool BoolVec::operator!=( const BoolVec &that ) const {
//    return ! operator ==( that );
//}

//bool BoolVec::operator<( const BoolVec &that ) const {
//    return size != that.size ? size < that.size : memcmp_bit( data, 0, that.data, 0, size ) < 0;

//}

//bool BoolVec::all_true() const {
//    if ( size ) {
//        size_t bs = size / 8;
//        if ( bs )
//            for( size_t i = 0; i < bs; ++i )
//                if ( data[ i ] != 255 )
//                    return false;
//        return size % 8 == 0 || PI8( data[ bs ] | ( 255 << size % 8 ) ) == 255;
//    }
//    return true;
//}

//bool BoolVec::all_false() const {
//    if ( size ) {
//        size_t bs = size / 8;
//        if ( bs )
//            for( size_t i = 0; i < bs; ++i )
//                if ( data[ i ] )
//                    return false;
//        return size % 8 == 0 || PI8( data[ bs ] & ~ ( 255 << size % 8 ) ) == 0;
//    }
//    return true;
//}

//bool BoolVec::all_true( size_t beg, size_t end ) const {
//    if ( beg % 8 )
//        TODO;

//    if ( size_t size = end - beg ) {
//        size_t bs = size / 8;
//        beg /= 8;
//        if ( bs )
//            for( size_t i = 0; i < bs; ++i )
//                if ( data[ beg + i ] != 255 )
//                    return false;
//        return size % 8 == 0 || PI8( data[ beg + bs ] | ( 255 << size % 8 ) ) == 255;
//    }
//    return true;
//}

//bool BoolVec::all_false( size_t beg, size_t end ) const {
//    if ( beg % 8 )
//        TODO;

//    if ( size_t size = end - beg ) {
//        size_t bs = size / 8;
//        beg /= 8;
//        if ( bs )
//            for( size_t i = 0; i < bs; ++i )
//                if ( data[ beg + i ] )
//                    return false;
//        return size % 8 == 0 || PI8( data[ beg + bs ] & ~ ( 255 << size % 8 ) ) == 0;
//    }
//    return true;
//}


//void BoolVec::set( size_t beg, size_t end, bool val ) {
//    PI8 *dst = data;
//    if ( size_t o = beg / 8 ) {
//        dst += o;
//        beg %= 8;
//    }

//    if ( val ) {
//        if ( beg ) {
//            TODO;
//        }
//        size_t len = end - beg;
//        for( ; len >= 8; len -= 8, ++dst )
//            *(PI8 *)dst = 0xFF;
//        if ( len )
//            *(PI8 *)dst |= ~( 0xFF << len );
//    } else {
//        if ( beg ) {
//            TODO;
//        }
//        size_t len = end - beg;
//        for( ; len >= 8; len -= 8, ++dst )
//            *(PI8 *)dst = 0x00;
//        if ( len )
//            *(PI8 *)dst &= 0xFF << len;
//    }
//}
