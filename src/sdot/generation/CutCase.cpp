#include "../support/TODO.h"
#include "CutCase.h"
#include <algorithm>

namespace sdot {

void CutCase::init( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &primitive_shapes ) {
    this->out_points = out_points;

    // 2D
    if ( rp.polytop.dim() == 2 )
        return _init_2D( rp, out_points, primitive_shapes );

    TODO;
}

void CutCase::_init_2D( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &primitive_shapes ) {
    std::vector<IndOut> points;
    for( TI i = 0; i < out_points.size(); ++i )
        points.push_back( { i, i, out_points[ i ] } );

    possibilities.push_back( std::make_unique<CutOpWithNamesAndInds>( rp.polytop.dim() ) );
    _init_2D_rec( *possibilities.back(), points, primitive_shapes );

    // remove void possibilities
    possibilities.erase( std::remove_if( possibilities.begin(), possibilities.end(), []( const CutOpWithNamesAndInds &p ) { return p.outputs.empty(); } ) );

    // set indices
    for( std::unique_ptr<CutOpWithNamesAndInds> &possibility : possibilities ) {
        for( TI inp_node = 0; inp_node < possibility->cut_op.nb_input_nodes(); ++inp_node )
            possibility->input_node_inds.push_back( inp_node );
        for( TI inp_face = 0; inp_face < possibility->cut_op.nb_input_faces(); ++inp_face )
            possibility->input_face_inds.push_back( inp_face );
    }
}

void CutCase::_init_2D_rec( CutOpWithNamesAndInds &possibility, const std::vector<IndOut> &points, const std::vector<NamedRecursivePolytop> &primitive_shapes ) {
    // find an in => out edge
    std::size_t io = points.size();
    for( TI i = 0; i < points.size(); ++i ) {
        TI j = ( i + 1 ) % points.size();
        if ( points[ i ].outside == false && points[ j ].outside ) {
            io = i;
            break;
        }
    }

    // => all outside ?
    if ( io == points.size() && ( points.empty() || points[ 0 ].outside ) )
        return;

    // => all inside ?
    if ( io == points.size() && points.size() && points[ 0 ].outside == false ) {
        CutOpWithNamesAndInds::Out output;
        output.shape_name = "S" + std::to_string( points.size() );
        for( TI i = 0; i < points.size(); ++i ) {
            output.output_node_inds.push_back( i );
            output.output_face_inds.push_back( i );
        }
        possibility.outputs.push_back( std::move( output ) );

        CutItem cut_item;
        for( const IndOut &p : points ) {
            cut_item.nodes.push_back( { p.ind_0, p.ind_1 } );
            cut_item.faces.push_back( p.ind_0 == p.ind_1 ? p.ind_0 : TI( -1 ) );
        }
        possibility.cut_op.cut_items.push_back( std::move( cut_item ) );
        return;
    }

    //
}

bool CutCase::_has_2D_shape( TI nb_points, const std::vector<NamedRecursivePolytop> &primitive_shapes ) {
    for( const NamedRecursivePolytop &ps : primitive_shapes )
        if ( ps.polytop.points.size() == nb_points )
            return true;
    return false;
}

}
