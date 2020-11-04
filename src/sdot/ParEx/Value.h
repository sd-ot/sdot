#pragma once

#include <vector>
#include <string>

namespace parex {
class Kernel;
class Task;

/**
  Essentially a wrapper around a `Task *`

  Can create
*/
class Value {
public:
    /**/               Value          ( const Kernel &kernel, std::vector<Value> &&input ); ///< value from a computation
    /**/               Value          ( const std::string &type, void *data ); ///< already known value (Task will take the ownership of *ptr)
    /**/               Value          ( const Value &that );
    template<class T>  Value          ( const T &value );
    /**/               Value          ( Value &&that );
    /**/               Value          ();

    /**/              ~Value          ();

    Value&             operator=      ( const Value &that );
    Value&             operator=      ( Value &&that );

    void               write_to_stream( std::ostream &os ) const;

    Task*              task;
};

inline Value data_from_value( std::int32_t  value ) { return { "std::int32_t" , new std::int32_t( value ) }; }
inline Value data_from_value( std::ostream* value ) { return { "std::ostream*", value }; }

template<class T>
Value::Value( const T &value ) : Value( data_from_value( value ) ) {
}


} // namespace parex
