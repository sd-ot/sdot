#include "ElementaryPolytopInfo.h"

ElementaryPolytopInfo::ElementaryPolytopInfo( const std::string &name ) : name( name ) {
    TI n = std::stoi( name );
    nodes = xt::arange( n );
}

typename ElementaryPolytopInfo::TI ElementaryPolytopInfo::nb_nodes() const {
    return nodes.size();
}

