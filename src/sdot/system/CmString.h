#pragma once

#include "TypeConfig.h"
#include <string.h>
#include <string>

namespace Hpipe {

/**
  To read or write data from reserved room in memory (data is not owned by CmQueue)
*/
class CmString {
public:
    CmString                  ( const void *beg, const void *end ) : beg( (const PI8 *)beg ), end( (const PI8 *)end ) {}
    CmString                  ( const void *beg, size_t len ) : beg( (const PI8 *)beg ), end( (const PI8 *)beg + len ) {}
    CmString                  ( const std::string &str ) : CmString( str.data(), str.size() ) {}
    CmString                  ( const char *beg ) : CmString( beg, strlen( beg ) ) {}
    CmString                  () : beg( 0 ), end( 0 ) {}

    CmString   slice          ( ST off, ST len ) { return { std::max( beg + off, beg ), std::min( beg + off + len, end ) }; }

    void       clear          () { beg = 0; end = 0; }

    operator   std::string    () const { return { beg, end }; }
    bool       operator==     ( const CmString &that ) const { return size() == that.size() && strncmp( (const char *)beg, (const char *)that.beg, size() ) == 0; }
    bool       operator==     ( const std::string &that ) const { return size() == that.size() && strncmp( (const char *)beg, that.data(), size() ) == 0; }
    bool       operator==     ( const char *that ) const { return size() == strlen( that ) && strncmp( (const char *)beg, that, size() ) == 0; }
    
    // error
    operator   bool           () const { return not error(); }
    bool       error          () const { return end == 0; } ///< works after at least a first read (and before free or clear)
    bool       ack_error      () { beg = 0; end = 0; return false; } ///< set error flag to true, and return false

    // size
    bool       empty          () const { return end == beg; }
    size_t     size           () const { return end - beg; }
    const PI8 *get_beg        () const { return beg; }
    const PI8 *get_end        () const { return end; }

    // readers. Beware there are no checks in these methods
    void       read_some      ( void *data, ST size ) { memcpy( data, beg, size ); beg += size; }
    void       skip_some      ( ST size ) { beg += size; }

    PI8        read_byte      () { return *( beg++ ); }
    const PI8 *ptr            () const { return beg; }

    void       set_ptr        ( const PI8 *ptr ) { if ( ptr >= beg and ptr <= end ) beg = ptr; else ack_error(); }

    // checkings for readers that save a signal (that will give error() != 0) if not ok. To be done before each read.
    bool       ack_read_byte  () { return beg < end ? true : ack_error(); } ///< return true if ok to read a byte. Else, set end to 0 (to signal an error) and return false.
    bool       ack_read_some  ( ST len ) { return beg + len <= end ? true : ack_error(); } ///< return true if ok to `len` bytes. Else, set end to 0 (to signal an error) and return false.

    // display
    void       write_to_stream( std::ostream &os ) const;

    bool       equal          ( const PI8 *ptr, ST len ) const { return end - beg == len and bcmp( beg, ptr, len ) == 0; }
    bool       equal          ( const char *ptr ) const { return equal( (const PI8 *)ptr, strlen( ptr ) ); }

protected:
    const PI8 *beg;
    const PI8 *end;
};

} // namespace Hpipe
