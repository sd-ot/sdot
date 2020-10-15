#include "CutCase.h"

namespace sdot {

void CutCase::init( const RecursivePolytop &/*rp*/, const std::vector<bool> &out_points ) {
    this->out_points = out_points;

    nb_new_edges = 0;
    for( TI i = 0; i < out_points.size(); ++i ) {
        TI j = ( i + 1 ) % out_points.size();
        nb_new_edges += out_points[ i ] == 0 && out_points[ j ] == 1;
    }

    // simple case: all_inside
    if ( std::find( out_points.begin(), out_points.end(), true ) == out_points.end() ) {
        CutItem cut_item;
        for( TI i = 0; i < out_points.size(); ++i )
            for( TI j = 0; j < 2; ++j )
                cut_item.node_inds.push_back( i );
        cownai.cut_op.cut_items.push_back( cut_item );

        cownai.outputs.push_back( { "shape_name", {} } );
        for( TI i = 0; i < out_points.size(); ++i )
            TODO;

        return;
    }


}

CutCase::TI CutCase::nb_created( std::string /*name*/ ) const {
    return cut_op.cut_items.size();
}

}
