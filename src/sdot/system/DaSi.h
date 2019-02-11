#pragma once

#include "TypeConfig.h"
// #include "WriteAs.h"
#include <iostream>
#include <limits>

namespace Hpipe {

///
struct DaSi {
    static constexpr PT npos = std::numeric_limits<PT>::max();

    DaSi( const char *data, PT size ) : data( data ), size( size ) {}
    DaSi( std::string &s ) : data( &s.front() ), size( s.size() ) {}
    DaSi() {}

    operator    bool           () const { return size != npos; }
    void        write_to_stream( std::ostream &os ) const;

    template<class Bq>
    void        write_to       ( Bq bq ) const { bq << size; bq.write_some( data, size ); }

    char        operator[]     ( PT i ) const { return data[ i ]; }

    char        front          () const { return data[ 0 ]; }
    char        back           () const { return data[ size - 1 ]; }

    PT          find           ( char c ) const; ///< return size if not found
    PT          rfind          ( char c ) const; ///< return size if not found

    DaSi        substr         ( PT beg, PT end ) const { return { data + beg, end - beg }; }
    DaSi        substr         ( PT end ) const { return { data, end }; }

    const char *beg            () const { return data; }
    const char *end            () const { return data + size; }

    const char *data;
    PT          size;
};

//template<class Bq,class CbString>
//void write_as( Bq bq, TypeList<CbString>, DaSi val ) { bq << val; }

} // namespace Hpipe
