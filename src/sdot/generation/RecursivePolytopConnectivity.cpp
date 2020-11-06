#include "../support/generic_ostream_output.h"
#include "RecursivePolytopConnectivity.h"

namespace parex {

RecursivePolytopConnectivity::RecursivePolytopConnectivity( TI nvi ) : nvi( nvi ) {
}

void RecursivePolytopConnectivity::write_to_stream( std::ostream &os ) const {
    os << "nvi=" << 2 << ":";
    for( TI node : nodes )
        os << " " << node;
    os << "\n";
}

} // namespace parex
