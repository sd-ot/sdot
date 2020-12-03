#pragma once

#include "Variant.h"

/**
*/
class VariantWithPtr : public Variant {
public:
    /**/     VariantWithPtr( const Type &type, void *data, bool owned = true );
    virtual ~VariantWithPtr();

    void    *data;         ///<
    bool     owned;        ///<
};

