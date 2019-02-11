#pragma once

#include "Stream.h"

/**
*/
class BoolVec {
public:
    BoolVec( Reference, void *_data, size_t _size ); ///<
    BoolVec( void *_data, size_t _size );            ///< makes a copy
    BoolVec( size_t _size, bool val );               ///< owned data
    BoolVec( const BoolVec &bv );                    ///< copy
    BoolVec( size_t _size = 0 );                     ///< owned non initialized data
    ~BoolVec();

    BoolVec &operator=      ( const BoolVec &that );

    BoolVec &operator|=     ( const BoolVec &that );
    BoolVec &operator&=     ( const BoolVec &that );

    bool     empty          () const { return _size == 0; }

    void     write_to_stream( std::ostream &os ) const;
    void     resize         ( size_t _size );

    bool     get            ( size_t index ) const { return _data[ index / nbw ] & ( W( 1 ) << index % nbw ); }            ///< get value at a given index

    void     set            ( size_t index ) { _data[ index / nbw ] |=   ( W( 1 ) << index % nbw ); }                      ///< assign true  to bit[ index ]
    void     clear          ( size_t index ) { _data[ index / nbw ] &= ~ ( W( 1 ) << index % nbw ); }                      ///< assign false to bit[ index ]

    void     set            ( size_t beg, size_t end ) { for( size_t index = beg; index < end; ++index ) set  ( index ); } ///< assign true  to bit[ beg .. end ]. TODO: optimize
    void     clear          ( size_t beg, size_t end ) { for( size_t index = beg; index < end; ++index ) clear( index ); } ///< assign false to bit[ beg .. end ]. TODO: optimize

    void     assign         ( size_t index, bool val ) { val ? set( index ) : clear( index ); }                            ///< assign val   to bit[ index ]

    void     set_all        () { for( size_t i = 0; i < ( _size + nbw - 1 ) / nbw; ++i ) _data[ i ] = ~ W( 0 ); }          ///< assign true  to all bits
    void     clear_all      () { for( size_t i = 0; i < ( _size + nbw - 1 ) / nbw; ++i ) _data[ i ] = 0; }                 ///< assign false to all bits

    size_t   size           () const { return _size; }

private:
    using W = std::uint64_t;
    enum { nbw = 8 * sizeof( W ) }; ///< nb bits per word

    W       *_data;
    size_t   _size;
    size_t   _rese;
};
