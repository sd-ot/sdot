#include "Internal/SubdividedIcosahedron.h"
#include "Internal/AreaOutput.h"
#include "ConvexPolyhedron3.h"
#include "Point2.h"

#ifdef PD_WANT_STAT
#include "../Support/Stat.h"
#endif

namespace sdot {

template<class Pc>
ConvexPolyhedron3<Pc>::ConvexPolyhedron3( const Tetra &tetra, CI cut_id ) : op_count( 0 ) {
    clear( tetra, cut_id );
}

template<class Pc>
ConvexPolyhedron3<Pc>::ConvexPolyhedron3( const Box &box, CI cut_id ) : op_count( 0 ) {
    clear( box, cut_id );
}

template<class Pc>
ConvexPolyhedron3<Pc>::ConvexPolyhedron3( ConvexPolyhedron3 &&cp ) :
    nb_connections( std::move( cp.nb_connections ) ),
    sphere_cut_id ( std::move( cp.sphere_cut_id  ) ),
    sphere_center ( std::move( cp.sphere_center  ) ),
    sphere_radius ( std::move( cp.sphere_radius  ) ),
    op_count      ( std::move( cp.op_count       ) ),
    faces         ( std::move( cp.faces          ) ),
    holes         ( std::move( cp.holes          ) ),
    edges         ( std::move( cp.edges          ) ),
    nodes         ( std::move( cp.nodes          ) ) {

    cp.sphere_radius = -1;

    if ( keep_min_max_coords ) {
        min_coord = cp.min_coord;
        max_coord = cp.max_coord;
    }
}

template<class Pc>
void ConvexPolyhedron3<Pc>::operator=( const ConvexPolyhedron3 &cp ) {
    if ( this == &cp )
        return;
    nb_connections = cp.nb_connections;
    sphere_cut_id  = cp.sphere_cut_id;
    sphere_center  = cp.sphere_center;
    sphere_radius  = cp.sphere_radius;
    op_count       = ++cp.op_count;

    if ( keep_min_max_coords ) {
        min_coord  = cp.min_coord;
        max_coord  = cp.max_coord;
    }

    faces.clear();
    holes.clear();
    edges.clear();
    nodes.clear();

    auto new_node = [&]( Node *orig ) -> Node * {
        if ( orig->op_count == cp.op_count )
            return orig->nitem.node;
        orig->op_count = cp.op_count;

        Node *node = add_node( orig->pos );
        if ( keep_min_max_coords )
            node->resp_bound = orig->resp_bound;
        orig->nitem.node = node;
        return node;
    };

    auto new_edge = [&]( Edge &orig ) -> Edge * {
        if ( orig.op_count == cp.op_count )
            return orig.nedge;
        orig.sibling->op_count = cp.op_count;
        orig.op_count = cp.op_count;

        if ( orig.round() ) {
            TODO;
        }

        EdgePair ep = add_straight_edge( new_node( orig.n0 ), new_node( orig.n1 ) );
        orig.sibling->nedge = ep.b;
        orig.nedge = ep.a;
        return ep.a;
    };

    for( const Face &orig : cp.faces ) {
        Face *face = faces.new_item();
        if ( allow_ball_cut)
            face->round = orig.round;
        face->op_count = 0;
        face->cut_id   = orig.cut_id;
        face->cut_O    = orig.cut_O;
        face->cut_N    = orig.cut_N;

        face->edges.clear();
        for( Edge &edge : orig.edges ) {
            Edge *nedge = new_edge( edge );
            face->edges.append( nedge );
            nedge->face = face;
        }
    }

    for( const Hole &orig : cp.holes ) {
        Hole *hole = holes.new_item();
        hole->cut_id = orig.cut_id;
        hole->cut_N  = orig.cut_N ;
        hole->cut_O  = orig.cut_O ;
    }
}

template<class Pc>
void ConvexPolyhedron3<Pc>::operator=( ConvexPolyhedron3 &&cp ) {
    sphere_cut_id  = std::move( cp.sphere_cut_id  );
    sphere_center  = std::move( cp.sphere_center  );
    sphere_radius  = std::move( cp.sphere_radius  );
    faces          = std::move( cp.faces          );
    holes          = std::move( cp.holes          );
    edges          = std::move( cp.edges          );
    nodes          = std::move( cp.nodes          );
    nb_connections = std::move( cp.nb_connections );
    op_count       = std::move( cp.op_count       );

    if ( keep_min_max_coords ) {
        min_coord  = cp.min_coord;
        max_coord  = cp.max_coord;
    }

    cp.sphere_radius = -1;
}

template<class Pc> template<class F>
bool ConvexPolyhedron3<Pc>::all_pos( const F &f ) const {
    for( const Node &node : nodes )
        if ( f( node.pos ) == false )
            return false;
    return true;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::intersect_with( const ConvexPolyhedron3 &cp ) {
    ASSERT( sphere_radius < 0 && cp.sphere_radius < 0, "TODO: intersect ball cutted with ball cutted convex polyhedron" );
    for( const Face &fp : cp.faces )
        plane_cut( fp.cut_O, fp.cut_N, fp.cut_id );
}

template<class Pc>
void ConvexPolyhedron3<Pc>::for_each_boundary_measure( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::ExpWmR2db<TF> &/*rf*/, const std::function<void(TF,CI)> &f, TF weight ) const {
    TODO;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::for_each_boundary_measure( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::WmR2 &/*rf*/, const std::function<void(TF,CI)> &f, TF weight ) const {
    TODO;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::for_each_boundary_measure( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::Unit &/*rf*/, const std::function<void(TF,CI)> &f, TF weight ) const {
    //    // round parts
    //    if ( flat_surfaces.empty() ) {
    //        if ( sphere_radius >= 0 )
    //            f( 4 * M_PI * std::pow( sphere_radius, 2 ), sphere_cut_id );
    //    } else if ( round_surfaces.size() == 1 ) {
    //        f( area( round_surfaces[ 0 ] ), sphere_cut_id );
    //    } else if ( round_surfaces.size() ) {
    //        // we substract area of the hole from area of the full sphere
    //        TF sa = 4 * M_PI * std::pow( sphere_radius, 2 );
    //        TF ar = sa * ( TF( 1 ) - nb_connections );
    //        for( const RoundSurface &rp : round_surfaces )
    //            ar += area( rp );
    //        f( ar, sphere_cut_id );
    //    }

    // flat parts
    for( const Face &fp : faces )
        f( sf.coeff * area( fp ), fp.cut_id );

    //    // holes
    //    for( const Hole &hole : holes ) {
    //        TF s = dot( hole.cut_O - sphere_center, hole.cut_N );
    //        TF r = std::sqrt( sphere_radius * sphere_radius - s * s );
    //        f( - 2 * M_PI * sphere_radius * ( sphere_radius - s ), sphere_cut_id );
    //        f( M_PI * r * r, hole.cut_id );
    //    }
}

template<class Pc>
void ConvexPolyhedron3<Pc>::for_each_boundary_measure( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::R2 &/*rf*/, const std::function<void(TF,CI)> &f, TF weight ) const {
    TODO;
}

template<class Pc> template<class Fu>
void ConvexPolyhedron3<Pc>::for_each_boundary_item( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::ExpWmR2db<TF> &/*rf*/, const Fu &f, TF weight ) const {
    TODO;
}

template<class Pc> template<class Fu>
void ConvexPolyhedron3<Pc>::for_each_boundary_item( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::WmR2 &/*rf*/, const Fu &f, TF weight ) const {
    TODO;
}

template<class Pc> template<class Fu>
void ConvexPolyhedron3<Pc>::for_each_boundary_item( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::Unit &/*rf*/, const Fu &f, TF weight ) const {
    TODO;
}

template<class Pc> template<class Fu>
void ConvexPolyhedron3<Pc>::for_each_boundary_item( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::R2 &/*rf*/, const Fu &f, TF weight ) const {
    TODO;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::write_to_stream(std::ostream &os) const {
    TODO;
    //    os << "nodes: " << nodes          << "\n";
    //    os << "edges: " << edges          << "\n";
    //    os << "round: " << round_surfaces << "\n";
    //    os << "flats: " << flat_surfaces  << "\n";
    //    os << "einds: " << edge_indices   << "\n";
    //    os << "holes: " << holes          << "\n";
}

template<class Pc>
void ConvexPolyhedron3<Pc>::display_html_canvas( std::ostream &os, TF weight, bool ext ) const {
    os << "TODO !\n";
}

template<class Pc>
void ConvexPolyhedron3<Pc>::ball_cut( Pt center, TF radius, CI cut_id ) {
    TODO;
    //    sphere_center = center;
    //    sphere_radius = radius;
    //    sphere_cut_id = cut_id;

    //    if ( cut_info.empty() ) {
    //        sphere_radius = 0;
    //        return;
    //    }

    //    bool all_inside = true;
    //    for( Node &node : nodes ) {
    //        node.soi.sd = norm_2_p2( node.pos - center ) - radius * radius;
    //        all_inside &= node.inside();
    //    }

    //    bool sphere_center_is_inside = true;
    //    for( const CutInfo &ci : cut_info ) {
    //        if ( dot( sphere_center - ci.cut_O, ci.cut_N ) >= 0 ) {
    //            sphere_center_is_inside = false;
    //            break;
    //        }
    //    }

    //    // if all points (corners) are inside the sphere, the sphere is not going to cut anything
    //    if ( all_inside )
    //        return;

    //    // cut edges
    //    TI old_nodes_size = nodes.size();
    //    TI old_edges_size = edges.size();
    //    edges.reserve( 2 * old_edges_size ); // we want to keep the references during the loop
    //    nodes.reserve( nodes.size() + old_edges_size );
    //    for( TI num_edge = 0; num_edge < old_edges_size; num_edge += 2 ) {
    //        Edge &edge_p0 = edges[ num_edge + 0 ];
    //        Edge &edge_p1 = edges[ num_edge + 1 ];
    //        Node &n0 = nodes[ edge_p0.n0 ];
    //        Node &n1 = nodes[ edge_p0.n1 ];

    //        auto find_unique_intersection = [&]( Pt p0, Pt p1 ) {
    //            // ( p0.x - sphere_center.x + ( p1.x - p0.x ) * t )² + ... = sphere_radius²
    //            TF a = norm_2_p2( p1 - p0 );
    //            TF b = dot( p0 - sphere_center, p1 - p0 );
    //            TF c = norm_2_p2( p0 - sphere_center ) - sphere_radius * sphere_radius;
    //            TF d = std::sqrt( std::max( TF( 0 ), b * b - a * c ) );
    //            TF u = ( - b + d ) / a;
    //            TF v = ( - b - d ) / a;
    //            TF t = std::abs( u - 0.5 ) <= std::abs( v - 0.5 ) ? u : v;
    //            return p0 + std::min( TF( 1 ), std::max( TF( 0 ), t ) ) * ( p1 - p0 );
    //        };

    //        auto find_two_cuts = [&]( Pt &pi0, Pt &pi1, const Pt &p0, const Pt &p1 ) {
    //            // ( p0.x - sphere_center.x + ( p1.x - p0.x ) * t )² + ... = sphere_radius²
    //            TF a = norm_2_p2( p1 - p0 );
    //            if ( a == 0 )
    //                return false;
    //            TF b = dot( p0 - sphere_center, p1 - p0 );
    //            TF c = norm_2_p2( p0 - sphere_center ) - sphere_radius * sphere_radius;
    //            TF s = b * b - a * c;
    //            if ( s <= 0 )
    //                return false;
    //            TF d = std::sqrt( s );
    //            TF u = ( - b - d ) / a;
    //            TF v = ( - b + d ) / a;
    //            if ( u > 0 && u < 1 )
    //                v = std::max( TF( 0 ), std::min( TF( 1 ), v ) );
    //            else if ( v > 0 && v < 1 )
    //                u = std::max( TF( 0 ), std::min( TF( 1 ), u ) );
    //            else
    //                return false;
    //            pi0 = p0 + u * ( p1 - p0 );
    //            pi1 = p0 + v * ( p1 - p0 );
    //            return true;
    //        };

    //        if ( s0 < 0 ) {
    //            if ( s1 < 0 ) {
    //                // no cut
    //                edge_p0.used = 1;
    //                edge_p1.used = 1;
    //            } else {
    //                TI nn = add_node( find_unique_intersection( p0, p1 ) );
    //                edge_p0.nedge = add_straight_edge( edge_p0.n0, nn, edge_p0.cut_index );
    //                edge_p1.nedge = edge_p0.nedge + 1;
    //                edge_p0.used = 0;
    //                edge_p1.used = 0;
    //            }
    //        } else {
    //            if ( s1 < 0 ) {
    //                TI nn = add_node( find_unique_intersection( p1, p0 ) );
    //                edge_p0.nedge = add_straight_edge( nn, edge_p0.n1, edge_p0.cut_index );
    //                edge_p1.nedge = edge_p0.nedge + 1;
    //                edge_p0.used = 0;
    //                edge_p1.used = 0;
    //            } else {
    //                // 2 or 0 cuts
    //                Pt Pi0, Pi1;
    //                if ( find_two_cuts( Pi0, Pi1, p0, p1 ) ) {
    //                    edge_p0.nedge = add_straight_edge( add_node( Pi0 ), add_node( Pi1 ), edge_p0.cut_index );
    //                    edge_p1.nedge = edge_p0.nedge + 1;
    //                } else {
    //                    edge_p0.nedge = TI( -1 );
    //                    edge_p1.nedge = TI( -1 );
    //                }
    //                edge_p0.used = 0;
    //                edge_p1.used = 0;
    //            }
    //        }
    //    }

    //    // update existing surfaces
    //    std::swap( edge_indices, old_edges_indices );
    //    TI first_cut_edge = edges.size();
    //    edge_indices.resize( 0 );
    //    for( TI num_flat_surface = 0; num_flat_surface < flat_surfaces.size(); ++num_flat_surface ) {
    //        FlatSurface &fs = flat_surfaces[ num_flat_surface ];
    //        TI new_beg_in_edge_indices = edge_indices.size();
    //        TI old_n1 = TI( -1 ), waiting_n0, waiting_ei = TI( -1 );

    //        edges.reserve( edges.size() + 2 * ( fs.end_in_edge_indices - fs.beg_in_edge_indices ) );
    //        for( TI num_in_edge_indices = fs.beg_in_edge_indices; num_in_edge_indices < fs.end_in_edge_indices; ++num_in_edge_indices ) {
    //            TI    num_edge = old_edges_indices[ num_in_edge_indices ];
    //            Edge &edge     = edges[ num_edge ];
    //            Node &n0       = nodes[ edge.n0 ];
    //            Node &n1       = nodes[ edge.n1 ];

    //            if ( s0 < 0 ) {
    //                if ( s1 < 0 ) {
    //                    edge_indices.push_back( num_edge );
    //                } else {
    //                    edge_indices.push_back( edge.nedge );
    //                    old_n1 = edges[ edge.nedge ].n1;
    //                }
    //            } else {
    //                if ( s1 < 0 ) {
    //                    if ( old_n1 != TI( -1 ) )
    //                        edge_indices.push_back( add_round_edge( old_n1, edges[ edge.nedge ].n0, fs.cut_index ) );
    //                    else {
    //                        waiting_n0 = edges[ edge.nedge ].n0;
    //                        waiting_ei = edge_indices.size();
    //                        edge_indices.push_back( 11700 );
    //                    }
    //                    edge_indices.push_back( edge.nedge );
    //                } else if ( edge.nedge != TI( -1 ) ) {
    //                    if ( old_n1 != TI( -1 ) )
    //                        edge_indices.push_back( add_round_edge( old_n1, edges[ edge.nedge ].n0, fs.cut_index ) );
    //                    else {
    //                        waiting_n0 = edges[ edge.nedge ].n0;
    //                        waiting_ei = edge_indices.size();
    //                        edge_indices.push_back( 11700 );
    //                    }

    //                    edge_indices.push_back( edge.nedge );

    //                    old_n1 = edges[ edge.nedge ].n1;
    //                }
    //            }
    //        }

    //        // if no remaining edges, remove the surface
    //        if ( new_beg_in_edge_indices == edge_indices.size() ) {
    //            // face cut ?
    //            TF dist = dot( cut_info[ fs.cut_index ].cut_O - sphere_center, cut_info[ fs.cut_index ].cut_N );
    //            if ( dist < sphere_radius && dist > -sphere_radius ) {
    //                Pt proj = sphere_center + dist * cut_info[ fs.cut_index ].cut_N;
    //                for( TI num_in_edge_indices = fs.beg_in_edge_indices; ; ++num_in_edge_indices ) {
    //                    if ( num_in_edge_indices == fs.end_in_edge_indices ) {
    //                        holes.push_back( { fs.cut_index } );
    //                        break;
    //                    }
    //                    Edge &edge = edges[ old_edges_indices[ num_in_edge_indices ] ];
    //                    Pt   &p0   = node_pos( edge.n0 );
    //                    Pt   &p1   = node_pos( edge.n1 );
    //                    if ( dot( cross_prod( proj - p0, p1 - p0 ), cut_info[ fs.cut_index ].cut_N ) < 0 )
    //                        break;
    //                }
    //            }

    //            // in all the case, remove the surface
    //            if ( num_flat_surface < flat_surfaces.size() - 1 )
    //                fs = flat_surfaces[ flat_surfaces.size() - 1 ];
    //            flat_surfaces.pop_back();
    //            --num_flat_surface;
    //        } else {
    //            // need to close the loop ?
    //            if ( waiting_ei != TI( -1 ) )
    //                edge_indices[ waiting_ei ] = add_round_edge( old_n1, waiting_n0, fs.cut_index );

    //            fs.beg_in_edge_indices = new_beg_in_edge_indices;
    //            fs.end_in_edge_indices = edge_indices.size();
    //        }
    //    }

    //    // add surfaces to cover the holes
    //    if ( first_cut_edge < edges.size() ) {
    //        RoundSurface rs;
    //        rs.beg_in_edge_indices = edge_indices.size();
    //        for( TI n = first_cut_edge + 1; n < edges.size(); n += 2 )
    //            edge_indices.push_back( n );

    //        TI old_n1 = edges[ edge_indices[ rs.beg_in_edge_indices ] ].n1;
    //        for( TI n = rs.beg_in_edge_indices + 1; n < edge_indices.size(); ++n ) {
    //            for( TI m = n; ; ++m ) {
    //                if ( m == edge_indices.size() ) {
    //                    rs.end_in_edge_indices = n;
    //                    part_round_surfaces.push_back( rs );

    //                    rs.beg_in_edge_indices = n;
    //                    old_n1 = edges[ edge_indices[ n ] ].n1;
    //                    break;
    //                }
    //                if ( edges[ edge_indices[ m ] ].n0 == old_n1 ) {
    //                    std::swap( edge_indices[ n ], edge_indices[ m ] );
    //                    old_n1 = edges[ edge_indices[ n ] ].n1;
    //                    break;
    //                }
    //            }
    //        }

    //        rs.end_in_edge_indices = edge_indices.size();
    //        part_round_surfaces.push_back( rs );
    //    }

    //    // remove unused items
    //    remove_unused_edges( old_edges_size, true );
    //    remove_unused_nodes( old_nodes_size );
    //    remove_unused_cuts ();

    //    //
    //    _make_ext_round_faces();

    //    //
    //    if ( cut_info.empty() && ! sphere_center_is_inside )
    //        sphere_radius = 0;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::for_each_node( const std::function<void( Pt v )> &f ) const {
    TODO;
}

template<class Pc> template<class F>
typename ConvexPolyhedron3<Pc>::Node *ConvexPolyhedron3<Pc>::find_node_maximizing( const F &f, bool return_node_only_if_true ) const {
    Node *node = faces.begin()->edges.begin()->n0;

    TF value;
    if ( f( value, node->pos ) )
        return node;

    while ( true ) {
        Node *best_node = node;
        TF best_value = value;
        for( const Edge &edge : node->edges ) {
            TF edge_n1_value;
            if ( f( edge_n1_value, edge.n1->pos ) )
                return edge.n1;
            if ( best_value < edge_n1_value ) {
                best_value = edge_n1_value;
                best_node = edge.n1;
            }
        }

        // nothing to raise the value ?
        if ( node == best_node )
            return return_node_only_if_true ? nullptr : node;

        value = best_value;
        node = best_node;
    }
}

template<class Pc>
void ConvexPolyhedron3<Pc>::mark_cut_faces_and_edges( MarkCutInfo &mci, Node *node, TF sp0 ) {
    // node is assumed to be exterior
    if ( node->op_count == op_count )
        return;
    node->op_count = op_count;
    mci.rem_nodes.append( node );

    if ( keep_min_max_coords )
        mci.mod_bounds |= node->resp_bound;

    for( Edge &edge : node->edges ) {
        if ( edge.op_count == op_count )
            continue;
        edge.sibling->op_count = op_count;
        edge.op_count = op_count;

        // mark faces
        auto mark_face = [&]( Face *face ) {
            if ( face->op_count == op_count )
                return;
            face->op_count = op_count;
            mci.cut_faces.append( face );
        };
        mark_face( edge.sibling->face );
        mark_face( edge.face );

        // node on the other side is exterior ?
        TF sp1 = dot( edge.n1->pos - mci.origin, mci.normal );
        if ( sp1 > 0 ) {
            // both nodes are out => we can remove the edge (the)
            mci.rem_edges.append( &edge );

            // we have a potentially new ext node to explore
            mark_cut_faces_and_edges( mci, edge.n1, sp1 );
        } else {
            // add a new node
            Node *nn = add_node( node->pos - sp0 / ( sp1 - sp0 ) * ( edge.n1->pos - node->pos ) );
            if ( keep_min_max_coords )
                nn->resp_bound = 0;
            nn->op_count = op_count;

            mci.cut_edges.append( &edge );
            edge.sibling->n1 = nn;
            edge.n0 = nn;
        }
    }
}

template<class Pc> template<int no>
void ConvexPolyhedron3<Pc>::plane_cut( Pt origin, Pt normal, CI cut_id, N<no> normal_is_normalized ) {
    // if void poly, there's nothing to do
    if ( faces.empty() )
        return;

    // find a node that has to be removed
    Node *node = find_node_maximizing( [&]( TF &criterion, const Pt &pos ) {
        criterion = dot( pos - origin, normal );
        return criterion > 0;
    } );
    if ( node == nullptr )
        return;

    // update normal if not already done
    if ( no == 0 )
        normal /= norm_2( normal );

    // update totally or partially the exterior edges
    ++op_count;
    MarkCutInfo mci;
    if ( keep_min_max_coords )
        mci.mod_bounds = 0;
    mci.origin = origin;
    mci.normal = normal;
    mark_cut_faces_and_edges( mci, node, dot( node->pos - origin, normal ) );

    #ifdef PD_WANT_STAT
    stat.add_for_dist( "nb outside during cut", mci.rem_nodes.size() );
    #endif

    // remove all the exterior nodes
    for( Node &node : mci.rem_nodes )
        nodes.free( &node );

    // register cut edges
    for( Edge &edge : mci.cut_edges )
        edge.n0->edges.append( &edge );

    // go through the faces which contain at least one ext node
    Node *last_created_node = nullptr;
    for( Face &face : mci.cut_faces ) {
        Edge *leave_ext = nullptr;
        Edge *enter_ext = nullptr;
        for( Edge &edge : face.edges ) {
            // n0 is exterior ?
            if ( edge.n0->op_count == op_count ) {
                if ( edge.n1->op_count != op_count ) {
                    if ( enter_ext ) {
                        EdgePair ep = add_straight_edge( enter_ext->n1, edge.n0 );
                        face.edges.insert_between( enter_ext, &edge, ep.a );

                        last_created_node = edge.n0;
                        ep.b->n0->nitem.edge = ep.b;
                        ep.a->face = &face;
                        break;
                    }
                    leave_ext = &edge;
                }
            } else {
                // n0 is inside, n1 node is outside
                if ( edge.n1->op_count == op_count ) {
                    if ( leave_ext ) {
                        EdgePair ep = add_straight_edge( edge.n1, leave_ext->n0 );
                        face.edges.set_front( leave_ext );
                        face.edges.set_back( &edge );
                        face.edges.append( ep.a );

                        last_created_node = edge.n1;
                        ep.b->n0->nitem.edge = ep.b;
                        ep.a->face = &face;
                        break;
                    }
                    enter_ext = &edge;
                }
            }
        }
        if ( leave_ext == nullptr && enter_ext == nullptr )
            faces.free( &face );
    }

    // remove all the exterior edges
    for( Edge &edge : mci.rem_edges ) {
        edges.free( edge.sibling );
        edges.free( &edge );
    }

    // make the new face
    if ( last_created_node ) {
        Face *face = faces.new_item();
        if ( allow_ball_cut )
            face->round = false;
        face->op_count = 0;
        face->cut_id   = cut_id;
        face->cut_O    = origin;
        face->cut_N    = normal;

        face->edges.clear();
        face->edges.append( last_created_node->nitem.edge );
        last_created_node->nitem.edge->face = face;
        for( Node *node = last_created_node->nitem.edge->n1; node != last_created_node; node = node->nitem.edge->n1 ) {
            if ( node == node->nitem.edge->n1 ) {
                //                 for( const Face &face : faces )
                //                     P( face.cut_O, face.cut_N );
                //                 P( origin, normal );
                TODO;
            }
            face->edges.append( node->nitem.edge );
            node->nitem.edge->face = face;
        }

        if ( keep_min_max_coords && mci.mod_bounds ) {
            for( int d = 0; d < dim; ++d ) {
                int mb0 = 1 << ( 2 * d + 0 );
                if ( mci.mod_bounds & mb0 ) {
                    min_coord[ d ] = + std::numeric_limits<TF>::max();
                    Node *resp_nodes = nullptr;
                    for( const Edge &edge : face->edges ) {
                        if ( min_coord[ d ] > edge.n0->pos[ d ] ) {
                            min_coord[ d ] = edge.n0->pos[ d ];
                            resp_nodes = edge.n0;
                        }
                    }
                    resp_nodes->resp_bound |= mb0;
                }

                int mb1 = 1 << ( 2 * d + 1 );
                if ( mci.mod_bounds & mb1 ) {
                    max_coord[ d ] = - std::numeric_limits<TF>::max();
                    Node *resp_nodes = nullptr;
                    for( const Edge &edge : face->edges ) {
                        if ( max_coord[ d ] < edge.n0->pos[ d ] ) {
                            max_coord[ d ] = edge.n0->pos[ d ];
                            resp_nodes = edge.n0;
                        }
                    }
                    resp_nodes->resp_bound |= mb1;
                }
            }
        }
    }
}

template<class Pc>
void ConvexPolyhedron3<Pc>::clear( const Tetra &tetra, CI cut_id ) {
    TODO;
    //    round_surfaces.resize( 0 );
    //    flat_surfaces      .resize( 0 );
    //    edge_indices       .resize( 0 );
    //    cut_info           .resize( 0 );
    //    edges              .resize( 0 );
    //    holes              .resize( 0 );
    //    pos                .resize( 4 * 2 * nb_glued_nodes );
    //    rpos               = pos.size() / 4;
    //    _nb_nodes          = 0;
    //    sphere_radius      = -1;


    //    // englobing tetra
    //    const TF cm = - std::sqrt( TF( 1 ) / 9 );
    //    const TF sm = + std::sqrt( TF( 8 ) / 9 );
    //    const TF qm = + std::sqrt( TF( 2 ) / 3 );

    //    const TI n0 = add_node( englobing_center + 4 * englobing_radius * Pt{  1,    0,        0 } );
    //    const TI n1 = add_node( englobing_center + 4 * englobing_radius * Pt{ cm,    0,     - sm } );
    //    const TI n2 = add_node( englobing_center + 4 * englobing_radius * Pt{ cm, + qm, 0.5 * sm } );
    //    const TI n3 = add_node( englobing_center + 4 * englobing_radius * Pt{ cm, - qm, 0.5 * sm } );

    //    const TI e0 = add_straight_edge( n0, n1, 0 );
    //    const TI e1 = add_straight_edge( n1, n2, 0 );
    //    const TI e2 = add_straight_edge( n2, n0, 0 );
    //    const TI e3 = add_straight_edge( n0, n3, 0 );
    //    const TI e4 = add_straight_edge( n3, n1, 0 );
    //    const TI e5 = add_straight_edge( n2, n3, 0 );

    //    auto add_face = [&]( TI e0, TI e1, TI e2 ) {
    //        const Pt &P0 = node_pos( edges[ e0 ].n0 );
    //        const Pt &P1 = node_pos( edges[ e0 ].n1 );
    //        const Pt &P2 = node_pos( edges[ e1 ].n1 );
    //        const Pt  n  = normalized( cross_prod( P0 - P1, P2 - P1 ) );
    //        const TI  ci = add_cut_info( node_pos( edges[ e0 ].n0 ), n, englobing_cut_id );
    //        add_flat_surface( add_edge_indices( e0, e1, e2 ), ci );
    //    };
    //    add_face( e2 + 1, e1 + 1, e0 + 1 );
    //    add_face( e0 + 0, e4 + 1, e3 + 1 );
    //    add_face( e5 + 0, e4 + 0, e1 + 0 );
    //    add_face( e2 + 0, e3 + 0, e5 + 1 );
    // if ( keep_min_max_coords )
    //     update_min_max_coord();
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::Node *ConvexPolyhedron3<Pc>::add_node( Pt pos ) {
    Node *res = nodes.new_item();
    res->op_count = 0;
    res->edges.clear();
    res->pos = pos;
    return res;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::clear( const Box &box, CI cut_id ) {
    sphere_radius = -1;
    faces.clear();
    holes.clear();
    edges.clear();
    nodes.clear();

    Node    *n0 = add_node( { box.p0.x, box.p0.y, box.p0.z } );
    Node    *n1 = add_node( { box.p1.x, box.p0.y, box.p0.z } );
    Node    *n2 = add_node( { box.p0.x, box.p1.y, box.p0.z } );
    Node    *n3 = add_node( { box.p1.x, box.p1.y, box.p0.z } );
    Node    *n4 = add_node( { box.p0.x, box.p0.y, box.p1.z } );
    Node    *n5 = add_node( { box.p1.x, box.p0.y, box.p1.z } );
    Node    *n6 = add_node( { box.p0.x, box.p1.y, box.p1.z } );
    Node    *n7 = add_node( { box.p1.x, box.p1.y, box.p1.z } );

    EdgePair e0 = add_straight_edge( n0, n1 );
    EdgePair e1 = add_straight_edge( n1, n3 );
    EdgePair e2 = add_straight_edge( n3, n2 );
    EdgePair e3 = add_straight_edge( n2, n0 );

    EdgePair e4 = add_straight_edge( n4, n6 );
    EdgePair e5 = add_straight_edge( n6, n7 );
    EdgePair e6 = add_straight_edge( n7, n5 );
    EdgePair e7 = add_straight_edge( n5, n4 );

    EdgePair e8 = add_straight_edge( n0, n4 );
    EdgePair e9 = add_straight_edge( n1, n5 );
    EdgePair ea = add_straight_edge( n3, n7 );
    EdgePair eb = add_straight_edge( n2, n6 );

    auto add_face = [&]( Edge *e0, Edge *e1, Edge *e2, Edge *e3 ) {
        const Pt &P0 = e0->n0->pos;
        const Pt &P1 = e0->n1->pos;
        const Pt &P2 = e1->n1->pos;

        Face *face = faces.new_item();
        if ( allow_ball_cut )
            face->round = false;
        face->op_count = 0;
        face->cut_id   = cut_id;
        face->cut_N    = normalized( cross_prod( P0 - P1, P2 - P1 ) );
        face->cut_O    = P0;

        face->edges.clear();
        face->edges.append( e0 );
        face->edges.append( e1 );
        face->edges.append( e2 );
        face->edges.append( e3 );

        e0->face = face;
        e1->face = face;
        e2->face = face;
        e3->face = face;
    };

    add_face( e0.a, e1.a, e2.a, e3.a );
    add_face( e4.a, e5.a, e6.a, e7.a );

    add_face( e8.a, e7.b, e9.b, e0.b );
    add_face( ea.a, e5.b, eb.b, e2.b );

    add_face( eb.a, e4.b, e8.b, e3.b );
    add_face( e9.a, e6.b, ea.b, e1.b );

    if ( keep_min_max_coords )
        update_min_max_coord();
}

template<class Pc>
void ConvexPolyhedron3<Pc>::add_centroid_contrib( Pt &ctd, TF &mea, const SpaceFunctions::Constant<TF> &/*sf*/, const FunctionEnum::ExpWmR2db<TF> &/*rf*/, TF weight ) const {
    TODO;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::add_centroid_contrib( Pt &ctd, TF &mea, const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::WmR2 &/*rf*/, TF weight ) const {
    TODO;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::add_centroid_contrib( Pt &ctd, TF &mea, const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::Unit &/*rf*/, TF weight ) const {
    // base
    if ( faces.empty() ) {
        TF vol = sf.coeff * 4 * M_PI / 3 * std::pow( std::max( TF( 0 ), sphere_radius ), 3 );
        ctd += vol * sphere_center;
        mea += vol;
    } else {
        Pt sc = sphere_radius < 0 ? faces.begin()->cut_O : sphere_center;
        for( const Face &face : faces ) {
            if ( allow_ball_cut && face.round ) {
                TODO;
                // Pt s_ctd;
                // TF s_mea;
                // _get_centroid_rf( s_ctd, s_mea );
                // TF vol = sphere_radius * s_mea / 3;
                // ctd += vol * ( sc + TF( 3 ) / 4 * ( s_ctd / s_mea - sc ) );
                // mea += vol;
            } else {
                Pt s_ctd;
                TF s_mea;
                _get_centroid_planar( s_ctd, s_mea, face );
                if ( s_mea ) {
                    TF sgd = dot( face.cut_O - sc, face.cut_N );
                    TF vol = sf.coeff * sgd * s_mea / 3;
                    ctd += vol * ( sc + TF( 3 ) / 4 * ( s_ctd / s_mea - sc ) );
                    mea += vol;
                }
            }
        }
    }

    // holes
    for( const Hole &hole : holes ) {
        TF h = sphere_radius - dot( hole.cut_O - sphere_center, hole.cut_N );
        TF c = 3 * std::pow( 2 * sphere_radius - h, 2 ) / ( 4 * ( 3 * sphere_radius - h ) );
        TF v = sf.coeff * M_PI / 3 * h * h * ( 3 * sphere_radius - h );
        ctd -= v * ( sphere_center + c * hole.cut_N );
        mea -= v;
    }
}

template<class Pc>
void ConvexPolyhedron3<Pc>::add_centroid_contrib( Pt &ctd, TF &mea, const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::R2 &/*rf*/, TF weight ) const {
    TODO;
}

template<class Pc>
bool ConvexPolyhedron3<Pc>::contains( const Pt &pos ) const {
    using std::pow;

    for( const Face &face : faces ) {
        if ( allow_ball_cut && face.round ) {
            if ( norm_2_p2( pos - sphere_center ) > pow( sphere_radius, 2 ) )
                return false;
        } else if ( dot( pos - face.cut_O, face.cut_N ) > 0 )
            return false;
    }

    return true;
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::distance( const Pt &pos, bool count_domain_boundaries ) const {
    using std::pow;
    using std::max;

    TF res = - std::numeric_limits<TF>::max();
    for( const Face &face : faces ) {
        if ( allow_ball_cut && face.round )
            res = max( res, norm_2_p2( pos - sphere_center ) - pow( sphere_radius, 2 ) );
        else if ( count_domain_boundaries || face.cut_id != -1ul )
            res = dot( pos - face.cut_O, face.cut_N );
    }

    return res;
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::Pt ConvexPolyhedron3<Pc>::min_position() const {
    Pt res{ + std::numeric_limits<TF>::max() };
    using std::min;
    for( const Node &node : nodes )
        res = min( res, node.pos );
    return res;
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::Pt ConvexPolyhedron3<Pc>::max_position() const {
    Pt res{ - std::numeric_limits<TF>::max() };
    using std::max;
    for( const Node &node : nodes )
        res = max( res, node.pos );
    return res;
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::Pt ConvexPolyhedron3<Pc>::centroid( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::Unit &rf ) const {
    TF mea = 0;
    Pt ctd = { 0, 0, 0 };
    add_centroid_contrib( ctd, mea, sf, rf );
    return mea ? ctd / mea : ctd;
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::integration_der_wrt_weight( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::ExpWmR2db<TF> &fu, TF weight ) const {
    return integration( sf, fu, weight ) / fu.eps;
}

template<class Pc> template<class FU>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::integration_der_wrt_weight( const SpaceFunctions::Constant<TF> &sf, const FU &, TF weight ) const {
    return 0;
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::integration( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::ExpWmR2db<TF> &/*rf*/, TF weight ) const {
    TODO;
    return 0;
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::integration( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::WmR2 &/*rf*/, TF weight ) const {
    TODO;
    return 0;
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::integration( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::Unit &/*rf*/, TF weight ) const {
    TF res;
    if ( faces.empty() ) {
        res = sphere_radius > 0 ? 4 * M_PI / 3 * std::pow( sphere_radius, 3 ) : 0;
    } else {
        res = 0;
        for( const Face &fp : faces ) {
            if ( allow_ball_cut && fp.round )
                TODO;
            res += dot( fp.cut_O, fp.cut_N ) * area( fp ) / 3;
        }

        //        if ( round_surfaces.size() == 1 ) {
        //            res = sphere_radius * area( round_surfaces[ 0 ] ) / 3;
        //        } else {
        //            // we substract area of the hole from area of the full sphere
        //            TF sa = 4 * M_PI * std::pow( sphere_radius, 2 );
        //            TF pa = sa * ( TF( 1 ) - nb_connections );
        //            for( const RoundSurface &rp : round_surfaces )
        //                pa += area( rp );
        //            res = sphere_radius * pa / 3;
        //        }

        //        const Pt &sc = sphere_center;
        //        for( const FlatSurface &fp : flat_surfaces )
        //            res += dot( cut_info[ fp.cut_index ].cut_O - sc, cut_info[ fp.cut_index ].cut_N ) * area( fp ) / 3;
    }

    for( const Hole &hole : holes ) {
        TF h = sphere_radius - dot( hole.cut_O - sphere_center, hole.cut_N );
        res -= M_PI / 3 * h * h * ( 3 * sphere_radius - h );
    }

    return sf.coeff * res;
}

template<class Pc> template<class F>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::gauss_integration( const F &func, int nb_gauss_points ) const {
    using std::abs;

    TF res = 0;
    if ( ! holes.empty() )
        TODO;

    Pt A = TF( 0 );
    for( const Node &node : nodes )
        A += node.pos;
    A /= TF( nodes.size() );

    for( const Face &fp : faces ) {
        if ( allow_ball_cut && fp.round )
            TODO;

        Pt B = TF( 0 );
        for( const Edge &edge : fp.edges )
            B += edge.n0->pos + edge.n1->pos;
        B /= TF( 2 * fp.edges.size() );

        for( const Edge &ed : fp.edges ) {
            const Pt &C = ed.n0->pos;
            const Pt &D = ed.n1->pos;
            Pt b = B - A;
            Pt c = C - A;
            Pt d = D - A;
            TF det = abs(
                b[ 0 ] * ( c[ 1 ] * d[ 2 ] - d[ 1 ] * c[ 2 ] ) +
                c[ 0 ] * ( d[ 1 ] * b[ 2 ] - b[ 1 ] * d[ 2 ] ) +
                d[ 0 ] * ( b[ 1 ] * c[ 2 ] - c[ 1 ] * b[ 2 ] )
            ) / 6;

            if ( nb_gauss_points <= 4 ) {
                const TF e = 0.5854101966249685;
                const TF f = 0.1381966011250105;
                res += det / 4 * (
                    func( f * A + e * B + e * C + e * D ) +
                    func( e * A + f * B + e * C + e * D ) +
                    func( e * A + e * B + f * C + e * D ) +
                    func( e * A + e * B + e * C + f * D )
                );
            } else if ( nb_gauss_points <= 11 ) {
                TF xa[] = { 0.2500000000000000, 0.7857142857142857, 0.0714285714285714, 0.0714285714285714, 0.0714285714285714, 0.1005964238332008, 0.3994035761667992, 0.3994035761667992, 0.3994035761667992, 0.1005964238332008, 0.1005964238332008 };
                TF ya[] = { 0.2500000000000000, 0.0714285714285714, 0.0714285714285714, 0.0714285714285714, 0.7857142857142857, 0.3994035761667992, 0.1005964238332008, 0.3994035761667992, 0.1005964238332008, 0.3994035761667992, 0.1005964238332008 };
                TF za[] = { 0.2500000000000000, 0.0714285714285714, 0.0714285714285714, 0.7857142857142857, 0.0714285714285714, 0.3994035761667992, 0.3994035761667992, 0.1005964238332008, 0.1005964238332008, 0.1005964238332008, 0.3994035761667992 };
                TF wt[] = {-0.0789333333333333, 0.0457333333333333, 0.0457333333333333, 0.0457333333333333, 0.0457333333333333, 0.1493333333333333, 0.1493333333333333, 0.1493333333333333, 0.1493333333333333, 0.1493333333333333, 0.1493333333333333 };
                for( int n = 0; n < 11; ++n ) {
                    TF aa = 1 - xa[ n ] - ya[ n ] - za[ n ];
                    res += det * wt[ n ] * func( aa * A + xa[ n ] * B + ya[ n ] * C + za[ n ] * D );
                }
            } else {
                TODO;
            }

        }
    }

    return res;
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::integration( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::R2 &/*rf*/, TF weight ) const {
    return sf.coeff * gauss_integration( [&]( const Pt &pt ) { return norm_2_p2( pt - sphere_center ); }, 11 );
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::Pt ConvexPolyhedron3<Pc>::random_point() const {
    TODO;
    return { TF( 0 ) };
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::boundary_measure( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::Unit &/*rf*/ ) const {
    using std::sqrt;
    using std::pow;
    using std::max;

    TF res;
    if ( faces.empty() ) {
        res = sphere_radius > 0 ? 4 * M_PI * pow( sphere_radius, 2 ) : 0;
    } else {
        res = 0;
        for( const Face &face : faces ) {
            if ( allow_ball_cut && face.round ) {
                // if ( round_surfaces.size() == 1 ) {
                //     res = area( round_surfaces[ 0 ] );
                // } else {
                //     // we substract area of the hole from area of the full sphere
                //     TF sa = 4 * M_PI * std::pow( sphere_radius, 2 );
                //     res = sa * ( TF( 1 ) - nb_connections );
                //     for( const RoundSurface &rp : round_surfaces )
                //         res += area( rp );
                // }
                TODO;
            } else
                res += area( face );
        }
    }

    for( const Hole &hole : holes ) {
        TF s = dot( hole.cut_O - sphere_center, hole.cut_N );
        TF r = sqrt( sphere_radius * sphere_radius - s * s );
        res += M_PI * ( r * r - 2 * sphere_radius * ( sphere_radius - s ) );
    }
    return sf.coeff * res;
}

template<class Pc> template<class Fu>
typename ConvexPolyhedron3<Pc>::Pt ConvexPolyhedron3<Pc>::centroid_ap( const Fu &fu, TI n ) const {
    TODO;
    return {};
    //    auto rdm1 = []() {
    //        return 2.0 * rand() / ( RAND_MAX - 1.0 ) - 1.0;
    //    };

    //    auto inside_semi_planes = [&]( Pt p ) {
    //        for( const CutInfo &ci : cut_info )
    //            if ( dot( p - ci.cut_O, ci.cut_N ) >= 0 )
    //                return false;
    //        return true;
    //    };

    //    TF count{ 0 };
    //    Pt centroid{ 0, 0, 0 };
    //    if ( sphere_radius < 0 ) {
    //        Pt mi = { + std::numeric_limits<TF>::max(), + std::numeric_limits<TF>::max(), + std::numeric_limits<TF>::max() };
    //        Pt ma = - mi;
    //        for( std::size_t n = 0; n < nb_points(); ++n ) {
    //            mi = min( mi, point( n ) );
    //            ma = max( ma, point( n ) );
    //        }
    //        for( TI i = 0; i < n; ++i ) {
    //            Pt p{ mi.x + ( ma.x - mi.x ) * rdm1(), mi.y + ( ma.y - mi.y ) * rdm1(), mi.z + ( ma.z - mi.z ) * rdm1() };
    //            if ( inside_semi_planes( p ) ) {
    //                centroid += fu( p, sphere_center ) * p;
    //                count += fu( p, sphere_center );
    //            }
    //        }
    //    } else {
    //        for( TI i = 0; i < n; ++i ) {
    //            Pt p{ rdm1(), rdm1(), rdm1() };
    //            if ( norm_2_p2( p ) <= 1 && inside_semi_planes( sphere_center + sphere_radius * p ) ) {
    //                centroid += fu( p, sphere_center ) * ( sphere_center + sphere_radius * p );
    //                count += fu( p, sphere_center );
    //            }
    //        }
    //    }

    //    return centroid / ( count + ( count == 0 ) );
}

template<class Pc> template<class Fu>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::measure_ap( const Fu &fu, TI n ) const {
    TODO;
    return 0;
    //    // width of the random distribution
    //    Pt sc = sphere_center;
    //    TF sr = sphere_radius;
    //    if ( sphere_radius < 0 ) {
    //        sc = { 0, 0, 0 };
    //        for( std::size_t n = 0; n < nb_points(); ++n )
    //            sc += point( n );
    //        sc /= TF( nb_points() );

    //        sr = 0;
    //        for( std::size_t n = 0; n < nb_points(); ++n )
    //            sr = std::max( sr, norm_2( point( n ) - sc ) );
    //    }

    //    auto rdm1 = []() {
    //        return 2.0 * rand() / ( RAND_MAX - 1.0 ) - 1.0;
    //    };

    //    auto inside_semi_planes = [&]( Pt p ) {
    //        for( const CutInfo &ci : cut_info )
    //            if ( dot( p - ci.cut_O, ci.cut_N ) >= 0 )
    //                return false;
    //        return true;
    //    };

    //    TF count = 0;
    //    for( TI i = 0; i < n; ++i ) {
    //        Pt p{ rdm1(), rdm1(), rdm1() };
    //        count += norm_2_p2( p ) <= 1 && inside_semi_planes( sc + sr * p );
    //    }

    //    return count * std::pow( 2 * sr, 3 ) / n;
}

template<class Pc> template<class Fu>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::boundary_measure_ap( const Fu &fu, TF max_ratio_area_error ) const {
    TODO;
    return 0;
    //    AreaOutput<Fu,TF> ao;
    //    display( ao, 0, true, max_ratio_area_error );
    //    return ao.area;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::update_min_max_coord() {
    for( int d = 0; d < dim; ++d ) {
        min_coord[ d ] = + std::numeric_limits<TF>::max();
        max_coord[ d ] = - std::numeric_limits<TF>::max();
    }

    std::array<Node *,6> resp_nodes;
    for( Node &node : nodes ) {
        TI num_resp = 0;
        node.resp_bound = 0;
        for( int d = 0; d < dim; ++d, num_resp += 2 ) {
            if ( min_coord[ d ] > node.pos[ d ] ) {
                resp_nodes[ num_resp + 0 ] = &node;
                min_coord[ d ] = node.pos[ d ];
            }
            if ( max_coord[ d ] < node.pos[ d ] ) {
                resp_nodes[ num_resp + 1 ] = &node;
                max_coord[ d ] = node.pos[ d ];
            }
        }
    }

    for( TI num_resp = 0; num_resp < resp_nodes.size(); ++num_resp )
        resp_nodes[ num_resp ]->resp_bound |= 1 << num_resp;
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::EdgePair ConvexPolyhedron3<Pc>::add_straight_edge( Node *n0, Node *n1 ) {
    Edge *a = edges.new_item();
    Edge *b = edges.new_item();

    n0->edges.append( a );
    n1->edges.append( b );

    a->sibling = b;
    b->sibling = a;

    a->op_count = 0;
    a->radius   = -1;
    a->n0       = n0;
    a->n1       = n1;

    b->op_count = 0;
    b->radius   = -1;
    b->n0       = n1;
    b->n1       = n0;

    return { a, b };
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TI ConvexPolyhedron3<Pc>::add_round_edge( Node *n0, Node *n1 ) {
    TODO;
    return 0;
    //    Edge edge;
    //    edge.n0        = n0;
    //    edge.n1        = n1;
    //    edge.cut_index = cut_index;
    //    edge.center    = sphere_center + dot( node_pos( n0 ) - sphere_center, cut_info[ cut_index ].cut_N ) * cut_info[ cut_index ].cut_N;
    //    edge.radius    = norm_2( node_pos( n0 ) - edge.center );

    //    // TODO: handle small edge radii
    //    if ( edge.radius ) {
    //        edge.X         = normalized( node_pos( n0 ) - edge.center );
    //        edge.tangent_0 = cross_prod( edge.X, cut_info[ cut_index ].cut_N );
    //        edge.tangent_1 = normalized( cross_prod( node_pos( n1 ) - edge.center, cut_info[ cut_index ].cut_N ) );
    //        edge.angle_1   = angle( edge, node_pos( n1 ) );
    //    } else {
    //        edge.X         = ortho_rand( cut_info[ cut_index ].cut_N );
    //        edge.tangent_0 = cross_prod( edge.X, cut_info[ cut_index ].cut_N );
    //        edge.tangent_1 = edge.tangent_0;
    //        edge.angle_1   = 0;
    //    }

    //    // positive version
    //    TI res = edges.size();
    //    edges.push_back( edge );

    //    // negative one
    //    std::swap( n0, n1 );
    //    edge.n0        = n0;
    //    edge.n1        = n1;
    //    // TODO: handle small edge radii
    //    if ( edge.radius ) {
    //        edge.X         = normalized( node_pos( n0 ) - edge.center );
    //        edge.tangent_0 = - cross_prod( edge.X, cut_info[ cut_index ].cut_N );
    //        edge.tangent_1 = - normalized( cross_prod( node_pos( n1 ) - edge.center, cut_info[ cut_index ].cut_N ) );
    //        edge.angle_1   = angle( edge, node_pos( n1 ) );
    //    } else {
    //        edge.X         = ortho_rand( cut_info[ cut_index ].cut_N );
    //        edge.tangent_0 = cross_prod( edge.X, cut_info[ cut_index ].cut_N );
    //        edge.tangent_1 = edge.tangent_0;
    //        edge.angle_1   = 0;
    //    }
    //    edges.push_back( edge );

    //    // return index on positive version
    //    return res;
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::angle( const Edge &edge, Pt p ) const {
    TODO;
    return 0;
    //    return atan2p( dot( p - edge.center, edge.Y() ), dot( p - edge.center, edge.X ) );
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::area( const Face &fs ) const {
    // round
    if ( allow_ball_cut && fs.round ) {
        TODO;
        //    TF res = 2 * M_PI;

        //    auto kg = [&]( const Edge &edge ) {
        //        TF s = dot( node_pos( edge.n0 ) - sphere_center, cut_info[ edge.cut_index ].cut_N ) > 0 ? -1 : +1;
        //        return s * std::sqrt( std::max( TF( 0 ), std::pow( sphere_radius, 2 ) - std::pow( edge.radius, 2 ) ) ) / sphere_radius * edge.angle_1;
        //    };

        //    for( TI i = rp.end_in_edge_indices - 1, j = rp.beg_in_edge_indices; j < rp.end_in_edge_indices; i = j++ )
        //        res -= std::acos( std::max( std::min( dot( edges[ edge_indices[ i ] ].tangent_1, edges[ edge_indices[ j ] ].tangent_0 ), TF( 1 ) ), TF( -1 ) ) );
        //    for( TI n = rp.beg_in_edge_indices; n < rp.end_in_edge_indices; ++n )
        //        res -= kg( edges[ edge_indices[ n ] ] );

        //    return std::pow( sphere_radius, 2 ) * res;
    }

    // area of straight triangles
    TF poly_area = 0;
    for( auto e0 = fs.edges.begin(), e1 = e0; ++e1 != fs.edges.end(); )
        poly_area += dot( cross_prod( e1->n1->pos - e0->n0->pos, e1->n0->pos - e0->n0->pos ), fs.cut_N );
    poly_area /= 2;

    // area of circular caps
    TF caps_area = 0;
    if ( allow_ball_cut ) {
        //        TODO;
        //        for( TI num_in_edge_indices = fs.beg_in_edge_indices; num_in_edge_indices < fs.end_in_edge_indices; ++num_in_edge_indices ) {
        //            const Edge &edge = edges[ edge_indices[ num_in_edge_indices ] ];
        //            if ( edge.round() ) {
        //                const Pt &Pi = node_pos( edge.n0 );
        //                const Pt &Pj = node_pos( edge.n1 );

        //                TF Ri = edge.radius;
        //                TF d0 = norm_2( Pj - Pi ) / 2;
        //                TF d1 = sqrt( std::max( TF( 0 ), std::pow( Ri, 2 ) - std::pow( d0, 2 ) ) );
        //                TF a1 = edge.angle_1;

        //                if ( a1 < M_PI )
        //                    caps_area -= d0 * d1;
        //                else
        //                    caps_area += d0 * d1;

        //                caps_area += a1 / 2 * std::pow( Ri, 2 );
        //            }
        //        }
    }

    return poly_area + caps_area;
}

//template<class Pc>
//void ConvexPolyhedron3<Pc>::_make_ext_round_faces() {
//    if ( part_round_surfaces.size() <= 1 )
//        return;

//    TODO;
//    //    // get node -> node links
//    //    for( Node &node : nodes )
//    //        node.soi.index = 0;

//    //    TI nb_con = 0;
//    //    for( TI i = 0; i < edges.size(); i += 2 ) {
//    //        ++nodes[ edges[ i ].n0 ].soi.index;
//    //        ++nodes[ edges[ i ].n1 ].soi.index;
//    //        nb_con += 2;
//    //    }
//    //    node_connectivity.resize( nb_con );

//    //    nb_con = 0;
//    //    for( Node &node : nodes ) {
//    //        TI lnb_con = node.soi.index;
//    //        node.soi.index = nb_con;
//    //        nb_con += lnb_con;
//    //    }

//    //    for( TI i = 0; i < edges.size(); i += 2 ) {
//    //       node_connectivity[ nodes[ edges[ i ].n0 ].soi.index++ ] = edges[ i ].n1;
//    //       node_connectivity[ nodes[ edges[ i ].n1 ].soi.index++ ] = edges[ i ].n0;
//    //    }

//    //    // get connected sets
//    //    num_connections.resize( nodes.size() );
//    //    for( TI i = nodes.size(); i--; )
//    //        num_connections[ i ] = TI( -1 );
//    //    nb_connections = TI( -1 );
//    //    for( TI i = nodes.size(); i--; ) {
//    //        if ( num_connections[ i ] == TI( -1 ) )
//    //            ++nb_connections;
//    //        _get_connections_rec( nb_connections, i );
//    //    }
//    //    ++nb_connections;
//}

template<class Pc>
void ConvexPolyhedron3<Pc>::_get_centroid_round( Pt &centroid, TF &area, const Face &fs ) const {
    TODO;

    // round
    //    centroid = { 0, 0, 0 };
    //    area = 0;
    //    for_each_triangle_rf( [&]( Pt P0, Pt P1, Pt P2 ) {
    //        TF a = norm_2( cross_prod( P1 - P0, P2 - P0 ) ) / 2;
    //        centroid += a / 3 * ( P0 + P1 + P2 );
    //        area += a;
    //    }, 1e-2, false );
}

template<class Pc>
void ConvexPolyhedron3<Pc>::_get_centroid_planar( Pt &centroid, TF &area, const Face &fs ) const {
    using std::sqrt;
    using std::max;
    using std::pow;
    using std::sin;
    using std::cos;

    centroid = { 0, 0, 0 };
    area = 0;

    // straight triangles
    const Pt &normal = fs.cut_N;
    auto edge_iter = fs.edges.begin();
    const Edge &e0 = *edge_iter;
    while( ++edge_iter != fs.edges.end() ) {
        const Edge &e1 = *edge_iter;
        Pt P0 = e0.n0->pos;
        Pt P1 = e1.n0->pos;
        Pt P2 = e1.n1->pos;

        TF a = dot( cross_prod( P2 - P0, P1 - P0 ), normal ) / 2;
        centroid += a / 3 * ( P0 + P1 + P2 );
        area += a;
    }

    // circular caps
    for( const Edge &edge : fs.edges ) {
        if ( edge.round() ) {
            const Pt &Pi = edge.n0->pos;
            const Pt &Pj = edge.n1->pos;

            TF Ri = edge.radius;
            TF d0 = norm_2( Pj - Pi ) / 2;
            TF d1 = sqrt( max( TF( 0 ), pow( Ri, 2 ) - pow( d0, 2 ) ) );
            TF a1 = edge.angle_1;
            TF ta = d1 * d0; // triangle area
            Pt tc = ta / 3 * ( edge.center + Pi + Pj ); // triangle centroid * area
            TF pa = a1 / 2 * pow( Ri, 2 ); // pie area
            Pt pc = pa * edge.center + 2.0 / 3.0 * pow( edge.radius, 3 ) * sin( a1 / 2 ) *
                    ( cos( a1 / 2 ) * edge.X() + sin( a1 / 2 ) * edge.Y() ); // pie centroid * area

            if ( a1 < M_PI ) {
                centroid += pc - tc;
                area += pa - ta;
            } else {
                centroid += pc + tc;
                area += pa + ta;
            }
        }
    }
}
template<class Pc> template<class V>
void ConvexPolyhedron3<Pc>::display( V &vo, const typename V::CV &cell_data, bool filled, TF max_ratio_area_error, bool display_tangents ) const {
    vo.mutex.lock();

    // round surfaces
    if ( filled ) {
        for_each_triangle_rf( [&]( Pt p0, Pt p1, Pt p2 ) {
            vo.add_polygon( { p0, p1, p2 }, cell_data );
        }, max_ratio_area_error, true );

        // flat surfaces
        for( const Face &face : faces ) {
            std::vector<Pt> points;
            for( const Edge &edge : face.edges )
                get_ap_edge_points( points, edge, max_ratio_area_error > 2e-2 ? 50 : 500 );
            vo.add_polygon( points, cell_data );
        }

        // hole planes
        for( const Hole &hole : holes ) {
            (void)hole;
            TODO;
            //            TF s = dot( cut_info[ hole.cut_index ].cut_O - sphere_center, cut_info[ hole.cut_index ].cut_N );
            //            Pt O = sphere_center + s * cut_info[ hole.cut_index ].cut_N;
            //            Pt X = ortho_rand( cut_info[ hole.cut_index ].cut_N );
            //            Pt Y = cross_prod( cut_info[ hole.cut_index ].cut_N, X );
            //            TF r = std::sqrt( sphere_radius * sphere_radius - s * s );
            //            std::vector<Pt> points;
            //            for( TI i = 0, d = max_ratio_area_error > 2e-2 ? 50 : 500; i < d; ++i )
            //                points.push_back( O + r * std::cos( i * 2 * M_PI / d ) * X + r * std::sin( i * 2 * M_PI / d ) * Y );
            //            vo.add_polygon( points, cell_data );
        }
    } else {
        for( const Face &face : faces ) {
            for( const Edge &edge : face.edges ) {
                if ( edge.n0 < edge.n1 ) {
                    std::vector<Pt> points;
                    get_ap_edge_points( points, edge, max_ratio_area_error > 2e-2 ? 50 : 500 );
                    vo.add_lines( points, cell_data );
                }
            }
        }
    }

    if ( display_tangents ) {
        TODO;
        //        for( TI i = 0; i < edges.size() / 2; ++i ) {
        //            const Edge &edge = edges[ 2 * i ];
        //            vo.add_arrow( node_pos( edge.n0 ), edge.tangent_0, cell_data );
        //            vo.add_arrow( node_pos( edge.n1 ), edge.tangent_1, cell_data );
        //        }
    }

    vo.mutex.unlock();
}

template<class Pc> template<class F>
void ConvexPolyhedron3<Pc>::for_each_triangle_rf( F &&func, TF max_ratio_area_error, bool remove_holes, std::mutex *m ) const {
    if ( sphere_radius <= 0 )
        return;

    TODO;
    //    // types
    //    using Triangle = typename SubdividedIcosahedron<TF>::Triangle;
    //    using Mesh     = typename SubdividedIcosahedron<TF>::Mesh;

    //    // start with a full mesh of the sphere
    //    static SubdividedIcosahedron<TF> si;
    //    static std::mutex m_si;
    //    m_si.lock();
    //    const Mesh &mesh = si.mesh_for_error( max_ratio_area_error );
    //    m_si.unlock();
    //    std::vector<Triangle> triangles = mesh.triangles;
    //    std::vector<Pt> points( mesh.points.size() );
    //    for( size_t i = 0; i < mesh.points.size(); ++i )
    //        points[ i ] = sphere_center + sphere_radius * mesh.points[ i ];

    //    // cut with edges
    //    if ( remove_holes ) {
    //        for( const CutInfo &ci : cut_info )
    //            p_cut( triangles, points, ci.cut_O, ci.cut_N );
    //    } else {
    //        for( const FlatSurface &fs : flat_surfaces )
    //            p_cut( triangles, points, cut_info[ fs.cut_index ].cut_O, cut_info[ fs.cut_index ].cut_N );
    //    }

    //    // display
    //    if ( m ) m->lock();
    //    for( const Triangle &triangle : triangles )
    //        func( points[ triangle.P0 ], points[ triangle.P1 ], points[ triangle.P2 ] );
    //    if ( m ) m->unlock();
}

template<class Pc>
void ConvexPolyhedron3<Pc>::get_ap_edge_points( std::vector<Pt> &points, const Edge &edge, int nb_divs, bool end ) const {
    if ( edge.straight() ) {
        points.push_back( edge.n0->pos );
        if ( end )
            points.push_back( edge.n1->pos );
    } else {
        for( int j = 0, n = std::ceil( edge.angle_1 * nb_divs / ( 2 * M_PI ) ); j < n + end; ++j )
            points.push_back( point_for_angle( edge, edge.angle_1 * j / n ) );
    }
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::Pt ConvexPolyhedron3<Pc>::point_for_angle( const Edge &edge, TF an ) const {
    using std::cos;
    using std::sin;
    return edge.center + edge.radius * cos( an ) * edge.X() + edge.radius * sin( an ) * edge.Y();
}

template<class Pc> template<class Triangle>
void ConvexPolyhedron3<Pc>::p_cut( std::vector<Triangle> &triangles, std::vector<Pt> &points, Pt cut_O, Pt cut_N ) {
    std::vector<Triangle> new_triangles;
    std::map<std::pair<TI,TI>,TI> edge_cuts;
    for( const Triangle &triangle : triangles ) {
        TI i0 = triangle.P0;
        TI i1 = triangle.P1;
        TI i2 = triangle.P2;

        Pt P0 = points[ i0 ];
        Pt P1 = points[ i1 ];
        Pt P2 = points[ i2 ];

        TF s0 = dot( P0 - cut_O, cut_N );
        TF s1 = dot( P1 - cut_O, cut_N );
        TF s2 = dot( P2 - cut_O, cut_N );

        // 1 * ( true if 0 is interior ) + 2 * ...
        switch( 1 * ( s0 < 0 ) + 2 * ( s1 < 0 ) + 4 * ( s2 < 0 ) ) {
        case 0 * 1 + 0 * 2 + 0 * 4:
            break;
        case 1 * 1 + 0 * 2 + 0 * 4: {
            Pt Q1 = P0 + s0 / ( s0 - s1 ) * ( P1 - P0 );
            Pt Q2 = P0 + s0 / ( s0 - s2 ) * ( P2 - P0 );
            TI M1 = _make_edge_cut( points, edge_cuts, i0, i1, Q1 );
            TI M2 = _make_edge_cut( points, edge_cuts, i0, i2, Q2 );
            new_triangles.push_back( { i0, M1, M2 } );
            break;
        }
        case 0 * 1 + 1 * 2 + 0 * 4: {
            Pt Q0 = P1 + s1 / ( s1 - s0 ) * ( P0 - P1 );
            Pt Q2 = P1 + s1 / ( s1 - s2 ) * ( P2 - P1 );
            TI M0 = _make_edge_cut( points, edge_cuts, i1, i0, Q0 );
            TI M2 = _make_edge_cut( points, edge_cuts, i1, i2, Q2 );
            new_triangles.push_back( { i1, M2, M0 } );
            break;
        }
        case 1 * 1 + 1 * 2 + 0 * 4: {
            Pt Q0 = P2 + s2 / ( s2 - s0 ) * ( P0 - P2 );
            Pt Q1 = P2 + s2 / ( s2 - s1 ) * ( P1 - P2 );
            TI M0 = _make_edge_cut( points, edge_cuts, i2, i0, Q0 );
            TI M1 = _make_edge_cut( points, edge_cuts, i2, i1, Q1 );
            new_triangles.push_back( { i0, i1, M1 } );
            new_triangles.push_back( { M1, M0, i0 } );
            break;
        }
        case 0 * 1 + 0 * 2 + 1 * 4: {
            Pt Q0 = P2 + s2 / ( s2 - s0 ) * ( P0 - P2 );
            Pt Q1 = P2 + s2 / ( s2 - s1 ) * ( P1 - P2 );
            TI M0 = _make_edge_cut( points, edge_cuts, i2, i0, Q0 );
            TI M1 = _make_edge_cut( points, edge_cuts, i2, i1, Q1 );
            new_triangles.push_back( { i2, M0, M1 } );
            break;
        }
        case 1 * 1 + 0 * 2 + 1 * 4: {
            Pt Q0 = P1 + s1 / ( s1 - s0 ) * ( P0 - P1 );
            Pt Q2 = P1 + s1 / ( s1 - s2 ) * ( P2 - P1 );
            TI M0 = _make_edge_cut( points, edge_cuts, i1, i0, Q0 );
            TI M2 = _make_edge_cut( points, edge_cuts, i1, i2, Q2 );
            new_triangles.push_back( { i0, M0, M2 } );
            new_triangles.push_back( { i0, M2, i2 } );
            break;
        }
        case 0 * 1 + 1 * 2 + 1 * 4: {
            Pt Q1 = P0 + s0 / ( s0 - s1 ) * ( P1 - P0 );
            Pt Q2 = P0 + s0 / ( s0 - s2 ) * ( P2 - P0 );
            TI M1 = _make_edge_cut( points, edge_cuts, i0, i1, Q1 );
            TI M2 = _make_edge_cut( points, edge_cuts, i0, i2, Q2 );
            new_triangles.push_back( { M1, i1, M2 } );
            new_triangles.push_back( { i1, i2, M2 } );
            break;
        }
        case 1 * 1 + 1 * 2 + 1 * 4: {
            new_triangles.push_back( { i0, i1, i2 } );
            break;
        }
        }
    }

    std::swap( triangles, new_triangles );
}

} // namespace sdot
