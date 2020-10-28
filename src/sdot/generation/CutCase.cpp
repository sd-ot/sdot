#include "../support/TODO.h"
#include "CutCase.h"

namespace sdot {

void CutCase::init( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &primitive_shapes ) {
    this->out_points = out_points;

    // simple case: all_outside
    if ( std::find( out_points.begin(), out_points.end(), false ) == out_points.end() )
        return;

    // simple case: all_inside
    if ( std::find( out_points.begin(), out_points.end(), true ) == out_points.end() ) {
        CutOpWithNamesAndInds possibility;

        CutOpWithNamesAndInds::Out output;
        output.shape_name = rp.name;
        CutItem cut_item;
        for( TI i = 0; i < out_points.size(); ++i ) {
            cut_item.nodes.push_back( { i, i } );
            possibility.input_node_inds.push_back( i );
            output.output_node_inds.push_back( i );
        }
        for( TI i = 0; i < rp.polytop.nb_faces(); ++i ) {
            cut_item.faces.push_back( i );
            possibility.input_face_inds.push_back( i );
            output.output_face_inds.push_back( i );
        }
        possibility.cut_op.cut_items.push_back( std::move( cut_item ) );
        possibility.cut_op.dim = rp.polytop.dim();
        possibility.outputs.push_back( output );

        possibilities.push_back( std::move( possibility ) );
        return;
    }

    // mixed case, 2D
    if ( rp.polytop.dim() == 2 )
        return _init_2D( rp, out_points, primitive_shapes );

    TODO;
}

void CutCase::_init_2D( const NamedRecursivePolytop &rp, const std::vector<bool> &out_points, const std::vector<NamedRecursivePolytop> &primitive_shapes ) {
    std::vector<bool> co = out_points;
    auto outside = [&]( std::size_t i ) { return co[ i ]; };
    auto inside = [&]( std::size_t i ) { return ! outside( i ); };

    while ( true ) {
        // find an in => out edge
        std::size_t io = co.size();
        for( TI i = 0; i < co.size(); ++i ) {
            TI j = ( i + 1 ) % co.size();
            if ( inside( i ) && outside( j ) ) {
                io = i;
                break;
            }
        }
        if ( io == co.size() )
            break;

        //
    }

}

}
