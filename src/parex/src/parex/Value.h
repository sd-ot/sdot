#pragma once

#include "type_name.h"
#include "SrcTask.h"
#include "Rc.h"

/**
  Essentially a wrapper around a `TaskRef`, with constructors for defined values and operator surdefinition

  For instance
  * `Value( 17 )` will create a TaskRef with known value `int( 17 )`
  * `Value( ... ) + Value( ... )` will call the gen_op(+) kernel
*/
class Value {
public:
    /**/                       Value          ( const Rc<Task> &task_ref );
    /**/                       Value          ( Rc<Task> &&task_ref );
    /**/                       Value          ( Task *task );

    /**/                       Value          ( const char *value );       ///< conversion to a std::string
    template<class T>          Value          ( T &&value );               ///< make a copy (or a move) of value
    template<class T>          Value          ( T *ptr, bool owned = true ); ///<

    /**/                       Value          ( const Value &that ) = delete;
    /**/                       Value          ( Value &&that ) = default;
    /**/                       Value          () = default;

    Value&                     operator=      ( const Value &that ) = delete;
    Value&                     operator=      ( Value &&that ) = default;

    void                       write_to_stream( std::ostream &os ) const;

    Value                      operator+      ( const Value &that ) const;
    Value                      operator-      ( const Value &that ) const;
    Value                      operator*      ( const Value &that ) const;
    Value                      operator/      ( const Value &that ) const;

    Value&                     operator+=     ( const Value &that );
    Value&                     operator-=     ( const Value &that );
    Value&                     operator*=     ( const Value &that );
    Value&                     operator/=     ( const Value &that );

    Rc<Task>                   to_string      ( double priority = 0 ) const;
    template<class T> Rc<Task> conv_to        () const { return conv_to( type_name( S<T>() ) ); }
    Rc<Task>                   conv_to        ( Type *type ) const;
    Rc<Task>                   conv_to        ( std::string type_name ) const;

    Rc<Task>                   task;
};

template<class T>
Value::Value( T *ptr, bool owned ) : Value( static_cast<Task *>( new SrcTask( Task::type_factory( type_name<T>() ), ptr, owned ) ) ) {
}

template<class T>
Value::Value( T &&value ) : Value( new typename std::decay<T>::type( std::forward<T>( value ) ), /*owned*/ true ) {
}
