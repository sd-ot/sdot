#pragma once

#include "RefCount.h"

/**
*/
template<class T>
struct TypedVariant {
    /**/      TypedVariant( RefCount *ref_count, T *data ) : ref_count( ref_count ), data( data ) {}
    /**/      TypedVariant( T *data ) : TypedVariant( new RefCount, data ) {}

    T*        operator->  () const { return data; }
    T&        operator*   () const { return *data; }

    RefCount* ref_count;  ///<
    T*        data;       ///<
};
