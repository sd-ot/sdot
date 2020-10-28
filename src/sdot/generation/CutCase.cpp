#include "../support/TODO.h"
#include "CutCase.h"

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
        points.push_back( { i, out_points[ i ] } );

    possibilities.push_back( std::make_unique<CutOpWithNamesAndInds>( rp.polytop.dim() ) );
    _init_2D_rec( *possibilities.back(), points, primitive_shapes );
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
        CutItem cut_item;

        for( TI i = 0; i < points.size(); ++i ) {
            cut_item.nodes.push_back( { i, i } );
            possibility.input_node_inds.push_back( i );
            output.output_node_inds.push_back( i );
        }

        for( TI i = 0; i < points.size(); ++i ) {
            cut_item.faces.push_back( i );
            possibility.input_face_inds.push_back( i );
            output.output_face_inds.push_back( i );
        }

        possibility.cut_op.cut_items.push_back( std::move( cut_item ) );
        possibility.outputs.push_back( output );

        possibilities.push_back( std::move( possibility ) );
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
