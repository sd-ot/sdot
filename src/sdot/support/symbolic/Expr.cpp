#include "Context.h"
#include "BinOp.h"

namespace Symbolic {

Expr::Expr( Inst *inst ) : inst( inst ) {
}

void Expr::write_to_stream( std::ostream &os ) const {
    inst->write_to_stream( os );
}

Expr bin_op( std::string name, Expr a, Expr b ) {
    Inst *ai = a.inst, *bi = b.inst;
    if ( ai > bi )
        std::swap( ai, bi );
    for( Inst *p : a.inst->parents )
        if ( BinOp *b = dynamic_cast<BinOp *>( p ) )
            if ( b->name == name && b->children[ 0 ] == ai && b->children[ 1 ] == bi )
                return p;
    return a.inst->context->pool.create<BinOp>( a.inst->context, name, ai, bi );
}

Expr operator+( Expr a, Expr b ) { return bin_op( "+", a, b ); }
Expr operator-( Expr a, Expr b ) { return bin_op( "-", a, b ); }
Expr operator*( Expr a, Expr b ) { return bin_op( "*", a, b ); }
Expr operator/( Expr a, Expr b ) { return bin_op( "/", a, b ); }

}
