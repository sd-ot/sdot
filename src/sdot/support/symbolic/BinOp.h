#pragma once

#include "Inst.h"
#include <string>

namespace Symbolic {

/** */
class BinOp : public Inst {
public:
    /***/        BinOp( Context *context, std::string name, Inst *a, Inst *b );

    virtual void write_to_stream( std::ostream &os ) const;

    std::string  name;
};

}
