#include "CutCase.h"

namespace sdot {

void CutCase::init( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &/*primitive_shapes*/ ) {
    this->out_points = out_points;

    nb_new_edges = 0;
    for( TI i = 0; i < out_points.size(); ++i ) {
        TI j = ( i + 1 ) % out_points.size();
        nb_new_edges += out_points[ i ] == 0 && out_points[ j ] == 1;
    }

    // cownai ======================================================================
    // simple case: all_inside
    if ( std::find( out_points.begin(), out_points.end(), true ) == out_points.end() ) {
        CutItem cut_item;
        for( TI i = 0; i < out_points.size(); ++i )
            for( TI j = 0; j < 2; ++j )
                cut_item.node_inds.push_back( i );
        cownai.cut_op.cut_items.push_back( cut_item );

        for( TI i = 0; i < out_points.size(); ++i )
            cownai.inputs.push_back( i );

        CutOpWithNamesAndInds::Out output;
        output.shape_name = rp.name;
        for( TI i = 0; i < out_points.size(); ++i )
            output.inds.push_back( i );
        cownai.outputs.push_back( output );

        return;
    }


}

CutCase::TI CutCase::nb_created( std::string /*name*/ ) const {
    return cownai.cut_op.cut_items.size();
}

}
