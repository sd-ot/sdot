#pragma once

#include "Scalar.h"

namespace parex {

/**
  A wrapper around a `Task`, with constructors and operators for Strings

  For instance
  * `Value( 17 )` will create a Task with known value `int( 17 )`
  * `Value( ... ) + Value( ... )` will call the gen_op(+) kernel
*/
class String : public TaskWrapper {
public:
    /**/          String    ( const std::string &str = {} );
    /**/          String    ( const char *str );
    /**/          String    ( Task *t );

    static String from_ptr  ( std::string *ptr, bool owned = true );

    Scalar        size      () const;

    String        operator+ ( const String &that ) const;

    String&       operator+=( const String &that );
};

} // namespace parex
