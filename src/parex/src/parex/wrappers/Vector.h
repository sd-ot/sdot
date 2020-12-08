#ifndef PAREX_Vector_H
#define PAREX_Vector_H

#include <initializer_list>
#include "Scalar.h"

namespace parex {

/**
  A wrapper around a `Task`, with constructors and operators for tensors
*/
class Vector : public TaskWrapper {
public:
    template<class T> Vector       ( std::initializer_list<T> &&l );
    /**/              Vector       ( Task *t );
    /**/              Vector       ();

    Scalar            size         () const;
};

} // namespace parex

#include "Vector.tcc"

#endif // PAREX_Vector_H
