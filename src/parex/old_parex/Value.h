#pragma once

#include "Task.h"

/**
  Essentially a wrapper around a `TaskRef`, with constructors for defined values and operator surdefinition

  For instance
  * `Value( 17 )` will create a TaskRef with known value `int( 17 )`
  * `Value( ... ) + Value( ... )` will call the gen_op(+) kernel
*/
class Value {
public:
    /**/              Value          ( const RcPtr<Task> &task_ref );
    /**/              Value          ( RcPtr<Task> &&task_ref );
    /**/              Value          ( Task *task );

    /**/              Value          ( const char *value );       ///< conversion to a std::string
    template<class T> Value          ( T &&value );               ///< make a copy (or a move) of value
    template<class T> Value          ( T *ptr, bool own = true ); ///<

    /**/              Value          ( const Value &that ) = delete;
    /**/              Value          ( Value &&that ) = default;
    /**/              Value          () = default;

    Value&            operator=      ( const Value &that ) = delete;
    Value&            operator=      ( Value &&that ) = default;

    void              write_to_stream( std::ostream &os ) const;

    Value             operator+      ( const Value &that ) const;
    Value             operator-      ( const Value &that ) const;
    Value             operator*      ( const Value &that ) const;
    Value             operator/      ( const Value &that ) const;

    Value&            operator+=     ( const Value &that );
    Value&            operator-=     ( const Value &that );
    Value&            operator*=     ( const Value &that );
    Value&            operator/=     ( const Value &that );

    RcPtr<Task>       task;
};

template<class T>
Value::Value( T &&value ) : Value( Task::from_data_ptr( new typename std::decay<T>::type( std::forward<T>( value ) ), true ) ) {
}

template<class T>
Value::Value( T *ptr, bool own ) : Value( Task::from_data_ptr( ptr, own ) ) {
}

