#include "../support/generic_ostream_output.h"
#include "RecursivePolytop.h"

namespace sdot {

RecursivePolytop::RecursivePolytop( TI nvi ) : connectivity( nvi ) {
}

RecursivePolytop RecursivePolytop::convex_hull( const std::vector<Pt> &points ) {
    RecursivePolytop res( points[ 0 ].size() );
    res.points = points;

    for( TI var = 0; var < points.size(); ++var )
        res.connectivity.nodes.push_back( var );

    return res;
}

void RecursivePolytop::write_to_stream( std::ostream &os ) const {
    os << "points:";
    for( const Pt &p : points  ) {
        os << " ";
        for( TI i = 0; i < p.size(); ++i )
            os << ( i ? "," : "" ) << p[ i ];
    }
    os << "\n";

    connectivity.write_to_stream( os );
}

RecursivePolytop::TI RecursivePolytop::nb_faces() const {
    return points.size();
}

RecursivePolytop::TI RecursivePolytop::dim() const {
    return points.size() ? points[ 0 ].size() : 0;
}

} // namespace sdot
