#pragma once

#include "../output_types/type_name.h"
#include "../task_graphs/SrcTask.h"
#include "TaskWrapper.h"

namespace parex {

/**
  A wrapper around a `Task`, with constructors and operators for scalars

  For instance
  * `Value( 17 )` will create a Task with known value `int( 17 )`
  * `Value( ... ) + Value( ... )` will call the gen_op(+) kernel
*/
class Scalar : public TaskWrapper {
public:
    using             TaskWrapper    ::TaskWrapper;
    /**/              Scalar         ( const char *str );           ///< conversion to a std::string
    template<class T> Scalar         ( T &&value );                 ///< make a copy (or a move) of value
    template<class T> Scalar         ( T *ptr, bool owned = true ); ///<

    void              write_to_stream( std::ostream &os ) const;

    Scalar            operator+      ( const Scalar &that ) const;
    Scalar            operator-      ( const Scalar &that ) const;
    Scalar            operator*      ( const Scalar &that ) const;
    Scalar            operator/      ( const Scalar &that ) const;

    Scalar&           operator+=     ( const Scalar &that );
    Scalar&           operator-=     ( const Scalar &that );
    Scalar&           operator*=     ( const Scalar &that );
    Scalar&           operator/=     ( const Scalar &that );
};

template<class T>
Scalar::Scalar( T *ptr, bool owned ) : TaskWrapper( new SrcTask( Task::type_factory( type_name<T>() ), ptr, owned ) ) {
}

template<class T>
Scalar::Scalar( T &&value ) : Scalar( new typename std::decay<T>::type( std::forward<T>( value ) ), /*owned*/ true ) {
}

} // namespace parex
