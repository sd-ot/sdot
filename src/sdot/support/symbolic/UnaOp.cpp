#include "UnaOp.h"

namespace Symbolic {

UnaOp::UnaOp( Context *context, std::string name, Inst *a ) : Inst( context ), name( name ) {
    a->parents.push_back( this );
    children = { a };
}

void UnaOp::write_to_stream( std::ostream &os ) const {
    children[ 0 ]->write_to_stream( os << name << "(" );
    os << ")";
}

void UnaOp::write_code( std::ostream &os ) const {
    os << name << " " << children[ 0 ]->reg;
}

}
