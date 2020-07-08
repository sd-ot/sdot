#pragma once

#include "Context.h"

namespace Symbolic {

/***/
class Codegen {
public:
    /**/                Codegen ( std::string TF = "TF", std::string sp = "  " );

    void                add_expr( std::string name, Expr expr );

    void                write   ( std::ostream &os );

private:
    struct              Output  { std::string name; Expr expr; };

    void                write   ( std::ostream &os, Inst *inst );

    std::vector<Output> outputs;
    std::size_t         nb_regs;
    std::string         TF;
    std::string         sp;
};

}

