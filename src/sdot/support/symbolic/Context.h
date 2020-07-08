#pragma once

#include "../BumpPointerPool.h"
#include "Number.h"
#include "Named.h"
#include "Inst.h"
#include "Expr.h"
#include <map>

namespace Symbolic {
class Codegen;

/**
*/
class Context {
public:
    /**/            Context  ();

    Expr            named    ( std::string name );
    Expr            number   ( Rational n );

private:
    using           Numbers  = std::map<Rational,Number *>;
    using           Nameds   = std::map<std::string,Named *>;
    friend Expr     bin_op   ( std::string name, Expr a, Expr b );
    friend Expr     una_op   ( std::string name, Expr a );
    friend class    Codegen;

    Numbers         numbers;
    Nameds          nameds;
    BumpPointerPool pool;
    std::size_t     date;
};

}
