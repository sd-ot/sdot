#include "CutCase.h"

namespace sdot {

void CutCase::init( const RecursivePolytop &rp, const std::vector<bool> &out_points ) {
    this->out_points = out_points;

    nb_new_edges = 0;
    for( TI i = 0; i < out_points.size(); ++i ) {
        TI j = ( i + 1 ) % out_points.size();
        nb_new_edges += out_points[ i ] == 0 && out_points[ j ] == 1;
    }
}

bool CutCase::all_inside() const {
    return std::find( out_points.begin(), out_points.end(), true ) == out_points.end();
}

CutCase::TI CutCase::nb_created( std::string /*name*/ ) const {
    if ( all_inside() )
        return 1;

    return 0;
}

}
