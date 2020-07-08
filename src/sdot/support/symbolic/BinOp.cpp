#include "BinOp.h"

namespace Symbolic {

BinOp::BinOp( Context *context, std::string name, Inst *a, Inst *b ) : Inst( context ), name( name ) {
    b->parents.push_back( this );
    a->parents.push_back( this );
    children = { a, b };
}

void BinOp::write_to_stream( std::ostream &os ) const {
    children[ 0 ]->write_to_stream( os << name << "(" );
    children[ 1 ]->write_to_stream( os << "," );
    os << ")";
}

void BinOp::write_code( std::ostream &os ) const {
    os << children[ 0 ]->reg << " " << name << " " << children[ 1 ]->reg;
}

}
