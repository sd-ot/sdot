#include "SymbolicElementaryPolytop.h"

SymbolicElementaryPolytop::SymbolicElementaryPolytop( const std::string &name ) : name( name ) {
    nodes = xt::arange( std::stoi( name ) );
}

unsigned SymbolicElementaryPolytop::nb_nodes() const {
    return nodes.size();
}

unsigned SymbolicElementaryPolytop::nb_faces() const {
    return nodes.size();
}
