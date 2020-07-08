#pragma once

#include "Inst.h"
#include <string>

namespace Symbolic {

/** */
class UnaOp : public Inst {
public:
    /***/        UnaOp( Context *context, std::string name, Inst *a );

    virtual void write_to_stream( std::ostream &os ) const;
    virtual void write_code     ( std::ostream &os ) const;

    std::string  name;
};

}
