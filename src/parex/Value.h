#pragma once

#include "TaskRef.h"

namespace parex {

/**
  Essentially a wrapper around a `TaskRef`, with constructors for
  * defined values. For instance, `Value( 17 )` will create a src task with known value `int( 17 )`
  * computations with a kernel. For instance, `Value( new Kernel(...), children )` will create a node in the computation graph

For instance,
*/
class Value {
public:
    /**/              Value          ( TaskRef &&task_ref );
    /**/              Value          ( Task *task );

    /**/              Value          ( const char *value );
    template<class T> Value          ( T &&value );
    template<class T> Value          ( T *ptr, bool own = true );

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

    TaskRef           ref;
};

template<class T>
Value::Value( T &&value ) : Value( Task::ref_on( new typename std::decay<T>::type( std::forward<T>( value ) ) ) ) {
}

template<class T>
Value::Value( T *ptr, bool own ) : Value( Task::ref_on( ptr, own ) ) {
}

} // namespace parex

