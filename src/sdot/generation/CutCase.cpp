#include "CutCase.h"

namespace sdot {

void CutCase::init( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &primitive_shapes ) {
    this->out_points = out_points;

    // nb_new_edges
    nb_new_edges = 0;
    for( TI i = 0; i < out_points.size(); ++i ) {
        TI j = ( i + 1 ) % out_points.size();
        nb_new_edges += out_points[ i ] == 0 && out_points[ j ] == 1;
    }

    // 2D case
    if ( rp.polytop.dim() == 2 )
        return init_2D( rp, out_points, primitive_shapes );

    // simple case: all_outside
    if ( std::find( out_points.begin(), out_points.end(), false ) == out_points.end() ) {
        return;
    }

    // simple case: all_inside
    if ( std::find( out_points.begin(), out_points.end(), true ) == out_points.end() ) {
        // op type
        CutItem cut_item;
        for( TI i = 0; i < out_points.size(); ++i )
            cut_item.nodes.push_back( { i, i } );
        for( TI i = 0; i < rp.polytop.nb_faces(); ++i )
            cut_item.faces.push_back( i );
        cownai.cut_op.cut_items.push_back( std::move( cut_item ) );

        // input indices
        for( TI i = 0; i < out_points.size(); ++i )
            cownai.input_node_inds.push_back( i );
        for( TI i = 0; i < rp.polytop.nb_faces(); ++i )
            cownai.input_face_inds.push_back( i );

        // output indices
        CutOpWithNamesAndInds::Out output;
        output.shape_name = rp.name;
        for( TI i = 0; i < out_points.size(); ++i )
            output.output_node_inds.push_back( i );
        for( TI i = 0; i < rp.polytop.nb_faces(); ++i )
            output.output_face_inds.push_back( i );
        cownai.outputs.push_back( output );

        return;
    }

    //
}

CutCase::TI CutCase::nb_created( std::string /*name*/ ) const {
    return cownai.cut_op.cut_items.size();
}

}
