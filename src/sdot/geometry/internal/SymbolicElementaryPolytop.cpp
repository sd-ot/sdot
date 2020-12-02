#include "SymbolicElementaryPolytop.h"
#include <sstream>

SymbolicElementaryPolytop::SymbolicElementaryPolytop( const std::string &name ) : name( name ) {
    nodes = xt::arange( std::stoi( name ) );
}

std::string SymbolicElementaryPolytop::vtk_output() const {
    if ( nodes.size() == 3 ) return "{ { 5, { 0, 1, 2 } } }";
    if ( nodes.size() == 4 ) return "{ { 9, { 0, 1, 3, 2 } } }";

    // poly
    std::ostringstream os;
    os << " { { 7, { ";
    for( std::size_t i = 0; i < nodes.size(); ++i )
        os << ( i ? ", " : "" ) << i;
    os << " } } }";
    return os.str();
}

unsigned SymbolicElementaryPolytop::nb_nodes() const {
    return nodes.size();
}

unsigned SymbolicElementaryPolytop::nb_faces() const {
    return nodes.size();
}