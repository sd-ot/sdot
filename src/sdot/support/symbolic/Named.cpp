#include "Named.h"

namespace Symbolic {

Named::Named( Context *context, std::string name ) : Inst( context ), name( name ) {
}

void Named::write_to_stream( std::ostream &os ) const {
    os << name;
}

void Named::write_code( std::ostream &os ) const {
    os << name;
}

}
