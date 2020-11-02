#include "../support/ASSERT.h"
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
        points.push_back( { i, i, i, out_points[ i ] } );

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
    // if everything is outside, there's nothing to do
    for( TI ni = 0;; ++ni ) {
        if ( ni == points.size() )
            return;
        if ( points[ ni ].outside == false )
            break;
    }

    // if everything is inside, add a cut_item with all the points
    for( TI ni = 0;; ++ni ) {
        if ( ni == points.size() ) {
            // if too many points, split in two parts
            if ( ! _has_2D_shape( points.size(), primitive_shapes ) ) {
                TI ns = points.size() / 2;
                std::vector<IndOut> new_points[ 2 ];
                for( TI nn = 0; nn < ns; ++nn )
                    new_points[ 0 ].push_back( { points[ nn ].ind_0, points[ nn ].ind_1, points[ nn ].face_id, false } );
                new_points[ 0 ].push_back( { points[ ns ].ind_0, points[ ns ].ind_1, TI( CutItem::internal_face ), false } );

                for( TI nn = ns; nn < points.size(); ++nn )
                    new_points[ 1 ].push_back( { points[ nn ].ind_0, points[ nn ].ind_1, points[ nn ].face_id, false } );
                new_points[ 1 ].push_back( { points[ 0 ].ind_0, points[ 0 ].ind_1, TI( CutItem::internal_face ), false } );

                for( TI z = 0; z < 2; ++z )
                    _init_2D_rec( possibility, new_points[ z ], primitive_shapes );
                return;
            }

            // else, simply add the shape
            CutOpWithNamesAndInds::Out output;
            output.shape_name = "S" + std::to_string( points.size() );
            possibility.outputs.push_back( output );

            CutItem cut_item;
            for( const IndOut &p : points ) {
                cut_item.nodes.push_back( { p.ind_0, p.ind_1 } );
                cut_item.faces.push_back( p.face_id );
            }

            // length of created edge
            for( TI i = 0; i < points.size(); ++i ) {
                if ( points[ i ].face_id == TI( CutItem::cut_face ) ) {
                    TI j = ( i + 1 ) % points.size();
                    cut_item.lengths.push_back( { std::array<TI,2>{ points[ i ].ind_0, points[ i ].ind_1 }, std::array<TI,2>{ points[ j ].ind_0, points[ j ].ind_1 } } );
                }
            }

            possibility.cut_op.cut_items.push_back( cut_item );
            return;
        }
        if ( points[ ni ].outside )
            break;
    }

    // we're going to cut, with eventually several possibilities
    // => find all the in -> out edges
    CutOpWithNamesAndInds cp_possibility = possibility;
    for( TI ni = 0, np = 0; ni < points.size(); ++ni ) {
        // can make a cut ?
        TI nj = ( ni + 1 ) % points.size();
        if ( points[ ni ].outside == false && points[ nj ].outside ) {
            // find all the out -> in edges
            for( TI nk = 0; nk < points.size(); ++nk ) {
                TI nl = ( nk + 1 ) % points.size();
                if ( points[ nk ].outside && points[ nl ].outside == false ) {
                    // call _init_2D_rec with the 2 parts
                    std::vector<IndOut> new_points[ 2 ];

                    // outside part
                    ASSERT( points[ ni ].plain(), "" );
                    ASSERT( points[ nj ].plain(), "" );
                    new_points[ 0 ].push_back( { points[ ni ].ind_0, points[ nj ].ind_0, points[ ni ].face_id, true } );
                    for( TI nn = nj; ; ++nn ) {
                        TI nm = nn % points.size();
                        new_points[ 0 ].push_back( { points[ nm ].ind_0, points[ nm ].ind_1, points[ nm ].face_id, points[ nm ].outside } );
                        if ( nm == nk )
                            break;
                    }
                    new_points[ 0 ].push_back( { points[ nk ].ind_0, points[ nl ].ind_0, TI( CutItem::cut_face ), true } );

                    if ( new_points[ 0 ].size() == 6 ) {
                        P( new_points[ 0 ].size() );
                        for( auto p : new_points[ 0 ] )
                            P( p.ind_0, p.ind_1 );
                    }

                    // inside part
                    ASSERT( points[ nk ].plain(), "" );
                    ASSERT( points[ nl ].plain(), "" );
                    new_points[ 1 ].push_back( { points[ nk ].ind_0, points[ nl ].ind_0, points[ nk ].face_id, false } );
                    for( TI nn = nl; ; ++nn ) {
                        TI nm = nn % points.size();
                        new_points[ 1 ].push_back( { points[ nm ].ind_0, points[ nm ].ind_1, points[ nm ].face_id, points[ nm ].outside } );
                        if ( nm == ni )
                            break;
                    }
                    new_points[ 1 ].push_back( { points[ ni ].ind_0, points[ nj ].ind_0, TI( CutItem::cut_face ), false } );

                    // recursion
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
