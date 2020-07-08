#include "Codegen.h"

namespace Symbolic {

Codegen::Codegen() {
}

void Codegen::add_expr( std::string name, Expr expr ) {
    outputs.push_back( { name, expr } );
}

void Codegen::write( std::ostream &os ) {
    os << "pouet";
}

}
