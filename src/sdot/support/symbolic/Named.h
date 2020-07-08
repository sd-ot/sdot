#pragma once

#include "Inst.h"
#include <string>

namespace Symbolic {

/** */
class Named : public Inst {
public:
    /***/        Named( Context *context, std::string name );

    virtual void write_to_stream( std::ostream &os ) const;

    std::string  name;
};

}
