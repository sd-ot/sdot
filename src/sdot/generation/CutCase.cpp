#include "../support/TODO.h"
#include "../support/P.h"
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

    // set indices
    for( std::unique_ptr<CutOpWithNamesAndInds> &possibility : possibilities ) {
        // input indices
        for( TI inp_node = 0; inp_node < possibility->cut_op.nb_input_nodes(); ++inp_node )
            possibility->input_node_inds.push_back( inp_node );
        for( TI inp_face = 0; inp_face < possibility->cut_op.nb_input_faces(); ++inp_face )
            possibility->input_face_inds.push_back( inp_face );
        // output indices
        for( TI n = 0; n < possibility->outputs.size(); ++n ) {
            CutOpWithNamesAndInds::Out &out = possibility->outputs[ n ];
            CutItem &cut_item = possibility->cut_op.cut_items[ n ];
            for( TI i = 0; i < cut_item.nodes.size(); ++i )
                out.output_node_inds.push_back( i );
            for( TI i = 0; i < cut_item.faces.size(); ++i )
                out.output_face_inds.push_back( i );
        }
    }

    // remove void possibilities
    possibilities.erase( std::remove_if( possibilities.begin(), possibilities.end(), []( const std::unique_ptr<CutOpWithNamesAndInds> &p ) {
        return p->outputs.empty();
    } ), possibilities.end() );
}

void CutCase::_init_2D_rec( CutOpWithNamesAndInds &possibility, const std::vector<IndOut> &points, const std::vector<NamedRecursivePolytop> &primitive_shapes ) {
    CutOpWithNamesAndInds cp_possibility = possibility;

    // find all the in -> out edges
    for( TI ni = 0, np = 0; ; ++ni ) {
        // all out or all in
        if ( ni == points.size() ) {
            // => all outside or no point ?
            if ( points.empty() || points[ 0 ].outside )
                return;

            // => all inside ?
            if ( ! _has_2D_shape( points.size(), primitive_shapes ) )
                return; // TODO

            // output shape name
            CutOpWithNamesAndInds::Out output;
            output.shape_name = "S" + std::to_string( points.size() );
            possibility.outputs.push_back( output );

            // add a cut_item with all the points
            CutItem cut_item;
            for( TI i = 0; i < points.size(); ++i ) {
                TI j = ( i + 1 ) % points.size();
                const IndOut &p = points[ i ], &q = points[ j ];

                cut_item.nodes.push_back( { p.ind_0, p.ind_1 } );

                if ( p.mid() && q.mid() )
                    cut_item.faces.push_back( TI( -1 ) );
                else
                    cut_item.faces.push_back( p.ind_0 );
            }
            possibility.cut_op.cut_items.push_back( cut_item );
            return;
        }

        // can make a cut
        TI nj = ( ni + 1 ) % points.size();
        if ( points[ ni ].outside == false && points[ nj ].outside ) {
            // find all the out -> in edges
            for( TI nk = 0; nk < points.size(); ++nk ) {
                TI nl = ( nk + 1 ) % points.size();
                if ( points[ nk ].outside && points[ nl ].outside == false ) {
                    // call _init_2D_rec with the 2 parts
                    std::vector<IndOut> new_points[ 2 ];

                    // outside part
                    new_points[ 0 ].push_back( { ni, nj, true } );
                    for( TI nn = nj; ; ++nn ) {
                        TI nm = nn % points.size();
                        new_points[ 0 ].push_back( { nm, nm, points[ nm ].outside } );
                        if ( nm == nk )
                            break;
                    }
                    new_points[ 0 ].push_back( { nk, nl, true } );

                    // inside part
                    new_points[ 1 ].push_back( { nk, nl, false } );
                    for( TI nn = nl; ; ++nn ) {
                        TI nm = nn % points.size();
                        new_points[ 1 ].push_back( { nm, nm, points[ nm ].outside } );
                        if ( nm == ni )
                            break;
                    }
                    new_points[ 1 ].push_back( { ni, nj, false } );

                    if ( np++ ) {
                        possibilities.push_back( std::make_unique<CutOpWithNamesAndInds>( cp_possibility ) );
                        CutOpWithNamesAndInds &new_possibility = *possibilities.back();
                        for( TI z = 0; z < 2; ++z )
                            _init_2D_rec( new_possibility, new_points[ z ], primitive_shapes );
                    } else {
                        for( TI z = 0; z < 2; ++z )
                            _init_2D_rec( possibility, new_points[ z ], primitive_shapes );
                    }
                }
            }
        }
    }
}

bool CutCase::_has_2D_shape( TI nb_points, const std::vector<NamedRecursivePolytop> &primitive_shapes ) {
    for( const NamedRecursivePolytop &ps : primitive_shapes )
        if ( ps.polytop.points.size() == nb_points )
            return true;
    return false;
}

}
