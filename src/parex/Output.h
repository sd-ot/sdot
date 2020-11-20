#pragma once

#include "OutputFuncList.h"
#include <utility>
#include <string>

namespace parex {

/**
*/
class Output {
public:
    using        Destroy  = void( void * );

    /**/         Output   ( std::size_t *ref_count, Destroy *destroy, const std::string &type, const void *data ) : ref_count( ref_count ), destroy( destroy ), type( type ), data( const_cast<void *>( data ) ) {}
    /**/         Output   () : ref_count( nullptr ), data( nullptr ) {}

    /**/         Output   ( const Output &that ) : ref_count( that.ref_count ), destroy( that.destroy ), type( that.type ), data( that.data ) { inc_ref(); }
    /**/         Output   ( Output &&that ) : ref_count( std::exchange( that.ref_count, nullptr ) ), destroy( that.destroy ), type( std::exchange( that.type, {} ) ), data( std::exchange( that.data, nullptr ) ) {}

    /**/        ~Output   () { dec_ref(); }

    Output&      operator=( const Output &that ) { that.inc_ref(); dec_ref(); ref_count = that.ref_count; destroy = that.destroy; type = that.type; data = that.data; return *this; }
    Output&      operator=( Output &&that ) { ref_count = std::exchange( that.ref_count, nullptr ); destroy = that.destroy; type = std::exchange( that.type, {} ); data = std::exchange( that.data, nullptr ); return *this; }

    // void      write_to_stream( std::ostream &os ) const { if ( task ) task->write_to_stream( os ); else os << "null"; os << "[" << nout << "]"; }

    void         inc_ref   () const { if ( ref_count ) ++ *ref_count; }
    void         dec_ref   () const { if ( ref_count && ! -- *ref_count ) { delete ref_count; destroy( data ); } }

    std::size_t* ref_count;
    Destroy*     destroy;
    std::string  type;
    void*        data;
};

} // namespace parex
