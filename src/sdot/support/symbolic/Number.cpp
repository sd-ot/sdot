#include "Number.h"

namespace Symbolic {

Number::Number( Context *context, Rational value ) : Inst( context ), value( value ) {
}

void Number::write_to_stream( std::ostream &os ) const {
    os << value;
}

void Number::write_code( std::ostream &os ) const {
    os << value;
}

}
