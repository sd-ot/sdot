#pragma once

#include "../BumpPointerPool.h"
#include "Named.h"
#include "Inst.h"
#include "Expr.h"
#include <map>

namespace Symbolic {

/**
*/
class Context {
public:
    Expr            named    ( std::string name );

private:
    using           Nameds   = std::map<std::string,Named *>;
    friend Expr     bin_op   ( std::string name, Expr a, Expr b );

    BumpPointerPool pool;
    Nameds          nameds;
};

}
