#include "../../src/sdot/support/ERROR.h"
#include "SimdGenOp.h"

SimdGenOp::SimdGenOp( Type type, ST len, std::string scalar_type ) : scalar_type( scalar_type ), type( type ), len( len ) {
}

bool SimdGenOp::commutative( Type type ) {
    return type == Type::Add || type == Type::Mul;
}

std::string SimdGenOp::str_op() const {
    switch ( type ) {
    case Type::Add: return "+";
    case Type::Div: return "/";
    case Type::Mul: return "*";
    case Type::Sub: return "-";
    default: ERROR( "" );
    }
}
