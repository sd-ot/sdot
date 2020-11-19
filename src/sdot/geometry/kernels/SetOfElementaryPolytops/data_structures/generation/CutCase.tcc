#include "../support/ASSERT.h"
#include "../support/TODO.h"
#include "../support/P.h"
#include "CutCase.h"
#include <algorithm>

inline void CutCase::init( const Element &rp, const std::vector<bool> &out_points, std::map<std::string,Element> &primitive_shapes ) {
    this->out_points = out_points;

    // 2D
    if ( rp.nvi == 2 )
        return _init_2D( rp, out_points, primitive_shapes );
    TODO;
}

inline void CutCase::_init_2D( const Element &rp, const std::vector<bool> &out_points, std::map<std::string,Element> &primitive_shapes ) {
    std::vector<IndOut> points;
    for( TI i = 0; i < out_points.size(); ++i )
        points.push_back( { i, i, i, out_points[ i ] } );

    possibilities.push_back( std::make_unique<CutOpWithNamesAndInds>( rp.nvi ) );
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

    // make unique
    if ( possibilities.size() > 1 ) {
        std::sort( possibilities.begin(), possibilities.end(), []( const std::unique_ptr<CutOpWithNamesAndInds> &a, const std::unique_ptr<CutOpWithNamesAndInds> &b ) {
            return a->created_shapes() < b->created_shapes();
        } );

        possibilities.erase( std::unique( possibilities.begin(), possibilities.end(), []( const std::unique_ptr<CutOpWithNamesAndInds> &a, const std::unique_ptr<CutOpWithNamesAndInds> &b ) {
                                return a->created_shapes() == b->created_shapes();
                            } ), possibilities.end() );
    }
}

inline void CutCase::_init_2D_rec( CutOpWithNamesAndInds &possibility, const std::vector<IndOut> &points, std::map<std::string,Element> &primitive_shapes ) {
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
            // start with the "lowest" point
            std::vector<IndOut> new_points = points;
            for( IndOut &io : new_points ) {
                TI a = std::min( io.ind_0, io.ind_1 );
                TI b = std::max( io.ind_0, io.ind_1 );
                io.ind_0 = a;
                io.ind_1 = b;
            }


            auto first = std::min_element( new_points.begin(), new_points.end() );
            std::rotate( new_points.begin(), first, new_points.end() );

            // if too many points, split in two parts
            if ( ! _has_2D_shape( new_points.size(), primitive_shapes ) ) {
                TI ns = new_points.size() / 2;
                std::vector<IndOut> spl_points[ 2 ];
                for( TI nn = 0; nn < ns; ++nn )
                    spl_points[ 0 ].push_back( { new_points[ nn ].ind_0, new_points[ nn ].ind_1, new_points[ nn ].face_id, false } );
                spl_points[ 0 ].push_back( { new_points[ ns ].ind_0, new_points[ ns ].ind_1, TI( CutItem::internal_face ), false } );

                for( TI nn = ns; nn < new_points.size(); ++nn )
                    spl_points[ 1 ].push_back( { new_points[ nn ].ind_0, new_points[ nn ].ind_1, new_points[ nn ].face_id, false } );
                spl_points[ 1 ].push_back( { new_points[ 0 ].ind_0, new_points[ 0 ].ind_1, TI( CutItem::internal_face ), false } );

                for( TI z = 0; z < 2; ++z )
                    _init_2D_rec( possibility, spl_points[ z ], primitive_shapes );
                return;
            }

            // else, simply add the shape
            CutOpWithNamesAndInds::Out output;
            output.shape_name = "S" + std::to_string( new_points.size() );
            possibility.outputs.push_back( output );

            CutItem cut_item;
            for( const IndOut &p : new_points ) {
                cut_item.nodes.push_back( { p.ind_0, p.ind_1 } );
                cut_item.faces.push_back( p.face_id );
            }

            // length of created edge
            for( TI i = 0; i < new_points.size(); ++i ) {
                if ( new_points[ i ].face_id == TI( CutItem::cut_face ) ) {
                    TI j = ( i + 1 ) % new_points.size();
                    cut_item.lengths.push_back( { std::array<TI,2>{ new_points[ i ].ind_0, new_points[ i ].ind_1 }, std::array<TI,2>{ new_points[ j ].ind_0, new_points[ j ].ind_1 } } );
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

inline bool CutCase::_has_2D_shape( int nb_points, std::map<std::string,Element> &primitive_shapes ) {
    for( const auto &ps : primitive_shapes )
        if ( ps.second.nb_nodes == nb_points )
            return true;
    return false;
}

inline bool CutCase::IndOut::operator<( const IndOut &that ) const {
    return std::tie( ind_0, ind_1, face_id, outside ) < std::tie( that.ind_0, that.ind_1, that.face_id, that.outside );
}

inline void CutCase::IndOut::write_to_stream( std::ostream &os ) const {
    os << ind_0 << "_" << ind_1 << "_" << face_id << "_" << ( outside ? 'o' : 'i' );
}

inline bool CutCase::IndOut::plain() const {
    return ind_0 == ind_1;
}
