#pragma once

#include "../utility/IsScalar.h"
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
    template      <class T,class = typename std::enable_if<IsScalar<T>::value>::type>
    /**/          Scalar    ( T &&value );                 ///< make a copy or get data from a value
    /**/          Scalar    ( Task *t );
    /**/          Scalar    ();                            ///< start with a Zero

    template      <class T,class = typename std::enable_if<IsScalar<T>::value>::type>
    static Scalar from_ptr  ( T *ptr, bool owned = true );

    Scalar        operator+ ( const Scalar &that ) const;
    Scalar        operator- ( const Scalar &that ) const;
    Scalar        operator* ( const Scalar &that ) const;
    Scalar        operator/ ( const Scalar &that ) const;

    Scalar&       operator+=( const Scalar &that );
    Scalar&       operator-=( const Scalar &that );
    Scalar&       operator*=( const Scalar &that );
    Scalar&       operator/=( const Scalar &that );
};

template<class T,class>
Scalar::Scalar( T &&value ) {
    using U = typename std::decay<T>::type;
    task = SrcTask::from_ptr( new U( std::forward<T>( value ) ), /*owned*/ true );
}

template<class T,class>
Scalar Scalar::from_ptr( T *ptr, bool owned ) {
    return SrcTask::from_ptr( ptr, owned );
}

} // namespace parex
