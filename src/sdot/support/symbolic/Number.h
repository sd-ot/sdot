#pragma once

#include "../Rational.h"
#include "Inst.h"

namespace Symbolic {

/** */
class Number : public Inst {
public:
    /***/        Number         ( Context *context, Rational value );

    virtual void write_to_stream( std::ostream &os ) const;
    virtual void write_code     ( std::ostream &os ) const;

    Rational     value;
};

}
