#pragma once

#include <ostream>

namespace Symbolic {
class Context;
class Inst;

/**
*/
class Expr {
public:
    /**/     Expr           ( Inst *inst );

    void     write_to_stream( std::ostream &os ) const;

    Inst*    inst;
};

Expr bin_op( std::string name, Expr a, Expr b );

Expr operator+( Expr a, Expr b );
Expr operator-( Expr a, Expr b );
Expr operator*( Expr a, Expr b );
Expr operator/( Expr a, Expr b );

}
