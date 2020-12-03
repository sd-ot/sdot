#pragma once

#include "RefCount.h"
#include "Type.h"

/**
*/
class Variant {
public:
    /**/     Variant    ( const Type &type );
    virtual ~Variant    ();

    RefCount ref_count; ///<
    Type     type;      ///<
};

