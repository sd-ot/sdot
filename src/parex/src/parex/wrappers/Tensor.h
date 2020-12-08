#ifndef PAREX_Tensor_H
#define PAREX_Tensor_H

#include "../resources/default_CpuAllocator.h"
#include "../containers/gtensor.h"
#include "../tasks/ListOfTask.h"
#include <initializer_list>
#include "String.h"
#include "Scalar.h"

namespace parex {

/**
  A wrapper around a `Task`, with constructors and operators for tensors
*/
class Tensor : public TaskWrapper {
public:
    template<class T> Tensor       ( std::initializer_list<std::initializer_list<std::initializer_list<T>>> &&l );
    template<class T> Tensor       ( std::initializer_list<std::initializer_list<T>> &&l );
    template<class T> Tensor       ( std::initializer_list<T> &&l );
    /**/              Tensor       ( Task *t );
    /**/              Tensor       ();

    static Tensor     from_function( std::function<void(Src &,SrcSet &)> &&code, const Scalar &dim, ListOfTask &&args = {}, const String &type = "parex::FP64", Memory *memory = default_CpuAllocator.memory() );
    static Tensor     from_function( const std::string &code, const Scalar &dim, ListOfTask &&args = {}, const String &type = "parex::FP64", Memory *memory = default_CpuAllocator.memory() );

    Tensor            operator+    ( const Tensor &that ) const;
    Tensor            operator-    ( const Tensor &that ) const;
    Tensor            operator*    ( const Tensor &that ) const;
    Tensor            operator/    ( const Tensor &that ) const;

    Tensor&           operator+=   ( const Tensor &that );
    Tensor&           operator-=   ( const Tensor &that );
    Tensor&           operator*=   ( const Tensor &that );
    Tensor&           operator/=   ( const Tensor &that );
};

} // namespace parex

#include "Tensor.tcc"

#endif // PAREX_Tensor_H
