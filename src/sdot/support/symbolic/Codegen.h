#pragma once

#include "Context.h"

namespace Symbolic {

/***/
class Codegen {
public:
    /**/                Codegen ();

    void                add_expr( std::string name, Expr expr );

    void                write   ( std::ostream &os );

private:
    struct              Output  { std::string name; Expr expr; };

    std::vector<Output> outputs;
};

}

