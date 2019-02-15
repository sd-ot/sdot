#include "SubdividedIcosahedron.h"
#include "ConvexPolyhedron3.h"
#include "AreaOutput.h"
#include "Point2.h"

template<class Pc>
ConvexPolyhedron3<Pc>::ConvexPolyhedron3( const Tetra &tetra, CI cut_id ) {
    clear( tetra, cut_id );
}

template<class Pc>
ConvexPolyhedron3<Pc>::ConvexPolyhedron3( const Box &box, CI cut_id ) {
    clear( box, cut_id );
}

template<class Pc>
void ConvexPolyhedron3<Pc>::for_each_boundary_measure( FunctionEnum::Unit, const std::function<void(TF,CI)> &f ) const {
    // round parts
    if ( flat_surfaces.empty() ) {
        if ( sphere_radius >= 0 )
            f( 4 * M_PI * std::pow( sphere_radius, 2 ), sphere_cut_id );
    } else if ( round_surfaces.size() == 1 ) {
        f( area( round_surfaces[ 0 ] ), sphere_cut_id );
    } else if ( round_surfaces.size() ) {
        // we substract area of the hole from area of the full sphere
        TF sa = 4 * M_PI * std::pow( sphere_radius, 2 );
        TF ar = sa * ( TF( 1 ) - nb_connections );
        for( const RoundSurface &rp : round_surfaces )
            ar += area( rp );
        f( ar, sphere_cut_id );
    }

    // flat parts
    for( const FlatSurface &fp : flat_surfaces )
        f( area( fp ), fp.cut_id );

    // holes
    for( const Hole &hole : holes ) {
        TF s = dot( hole.cut_O - sphere_center, hole.cut_N );
        TF r = std::sqrt( sphere_radius * sphere_radius - s * s );
        f( - 2 * M_PI * sphere_radius * ( sphere_radius - s ), sphere_cut_id );
        f( M_PI * r * r, hole.cut_id );
    }
}

template<class Pc>
void ConvexPolyhedron3<Pc>::write_to_stream(std::ostream &os) const {
    os << "nodes: " << nodes          << "\n";
    os << "edges: " << edges          << "\n";
    os << "round: " << round_surfaces << "\n";
    os << "flats: " << flat_surfaces  << "\n";
    os << "einds: " << edge_indices   << "\n";
    os << "holes: " << holes          << "\n";
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

template<class Pc> template<int no>
void ConvexPolyhedron3<Pc>::plane_cut( Pt origin, Pt normal, CI cut_id, N<no> normal_is_normalized ) {
    bool all_ko = true;
    bool all_ok = true;
    nodes.for_each_offset( [&]( TN n ) {
        TF s = dot( nodes.pos( n ) - origin, normal );
        nodes.set_sp( n, s );
        bool ok = s < 0;
        all_ko &= ! ok;
        all_ok &= ok;
    } );

    if ( all_ok )
        return;

    if ( all_ko ) {
        flat_surfaces.clear();
        edges        .clear();
        nodes        .clear();
        return;
    }

    if ( no == 0 )
        normal /= norm_2( normal );

    // new nodes and new edges. edge.nedge will contain index of the new edges. edge.used will be != 0 if edge is kept
    TI old_nodes_size = _nb_nodes;
    TI old_edges_size = edges.size();
    edges.reserve( 2 * old_edges_size ); // we want to keep the references during the loop
    reserve_nodes( _nb_nodes + old_edges_size );
    for( TI num_edge = 0; num_edge < old_edges_size; num_edge += 2 ) {
        Edge &edge_p0 = edges[ num_edge + 0 ];
        Edge &edge_p1 = edges[ num_edge + 1 ];
        Pt p0 = node_pos( edge_p0.n0 );
        Pt p1 = node_pos( edge_p0.n1 );
        TF s0 = node_sp( edge_p0.n0 );
        TF s1 = node_sp( edge_p0.n1 );

        if ( s0 < 0 ) {
            if ( s1 < 0 ) {
                edge_p0.used = 1;
                edge_p1.used = 1;
            } else {
                TI nn = add_node( p0 - s0 / ( s1 - s0 ) * ( p1 - p0 ) );
                edge_p0.nedge = add_straight_edge( edge_p0.n0, nn, edge_p0.cut_index );
                edge_p1.nedge = edge_p0.nedge + 1;
                edge_p0.used = 0;
                edge_p1.used = 0;
            }
        } else {
            if ( s1 < 0 ) {
                TI nn = add_node( p0 + s0 / ( s0 - s1 ) * ( p1 - p0 ) );
                edge_p0.nedge = add_straight_edge( nn, edge_p0.n1, edge_p0.cut_index );
                edge_p1.nedge = edge_p0.nedge + 1;
                edge_p0.used = 0;
                edge_p1.used = 0;
            } else {
                edge_p0.used = 0;
                edge_p1.used = 0;
            }
        }
    }

    // update existing surfaces
    std::swap( edge_indices, old_edges_indices );
    TI first_cut_edge = edges.size();
    edge_indices.resize( 0 );
    for( TI num_flat_surface = 0; num_flat_surface < flat_surfaces.size(); ++num_flat_surface ) {
        FlatSurface &fs = flat_surfaces[ num_flat_surface ];
        TI new_beg_in_edge_indices = edge_indices.size();
        TI old_n1 = TI( -1 ), waiting_n0, waiting_ei = TI( -1 );

        edges.reserve( edges.size() + 2 * ( fs.end_in_edge_indices - fs.beg_in_edge_indices ) );
        for( TI num_in_edge_indices = fs.beg_in_edge_indices; num_in_edge_indices < fs.end_in_edge_indices; ++num_in_edge_indices ) {
            TI    num_edge = old_edges_indices[ num_in_edge_indices ];
            Edge &edge     = edges[ num_edge ];
            TF    s0       = node_sp( edge.n0 );
            TF    s1       = node_sp( edge.n1 );

            if ( s0 < 0 ) {
                if ( s1 < 0 ) {
                    edge_indices.push_back( num_edge );
                } else {
                    edge_indices.push_back( edge.nedge );
                    old_n1 = edges[ edge.nedge ].n1;
                }
            } else if ( s1 < 0 ) {
                if ( old_n1 != TI( -1 ) )
                    edge_indices.push_back( add_straight_edge( old_n1, edges[ edge.nedge ].n0, fs.cut_index ) );
                else {
                    waiting_n0 = edges[ edge.nedge ].n0;
                    waiting_ei = edge_indices.size();
                    edge_indices.push_back( 11700 );
                }

                edge_indices.push_back( edge.nedge );
            }
        }

        // if no remaining edges, remove the surface
        if ( new_beg_in_edge_indices == edge_indices.size() ) {
            if ( num_flat_surface < flat_surfaces.size() - 1 )
                fs = flat_surfaces[ flat_surfaces.size() - 1 ];
            flat_surfaces.pop_back();
            --num_flat_surface;
        } else {
            // need to close the loop ?
            if ( waiting_ei != TI( -1 ) )
                edge_indices[ waiting_ei ] = add_straight_edge( old_n1, waiting_n0, fs.cut_index );

            fs.beg_in_edge_indices = new_beg_in_edge_indices;
            fs.end_in_edge_indices = edge_indices.size();
        }
    }

    // add a surface to cover the hole
    if ( first_cut_edge < edges.size() ) {
        FlatSurface fs;
        fs.beg_in_edge_indices = edge_indices.size();
        for( TI n = first_cut_edge + 1; n < edges.size(); n += 2 )
            edge_indices.push_back( n );

        TI old_n1 = edges[ edge_indices[ fs.beg_in_edge_indices ] ].n1;
        for( TI n = fs.beg_in_edge_indices + 1; n < edge_indices.size(); ++n ) {
            for( TI m = n; m < edge_indices.size(); ++m ) {
                if ( edges[ edge_indices[ m ] ].n0 == old_n1 ) {
                    std::swap( edge_indices[ n ], edge_indices[ m ] );
                    old_n1 = edges[ edge_indices[ n ] ].n1;
                    break;
                }
            }
        }

        fs.end_in_edge_indices = edge_indices.size();
        fs.cut_index           = cut_index;
        flat_surfaces.push_back( fs );
    }

    // remove unused items
    remove_unused_edges( old_edges_size, false );
    remove_unused_nodes( old_nodes_size );
    remove_unused_cuts ();
}

template<class Pc>
void ConvexPolyhedron3<Pc>::clear( Pt englobing_center, TF englobing_radius, CI englobing_cut_id ) {
    round_surfaces.resize( 0 );
    flat_surfaces      .resize( 0 );
    edge_indices       .resize( 0 );
    cut_info           .resize( 0 );
    edges              .resize( 0 );
    holes              .resize( 0 );
    pos                .resize( 4 * 2 * nb_glued_nodes );
    rpos               = pos.size() / 4;
    _nb_nodes          = 0;
    sphere_radius      = -1;


    // englobing tetra
    const TF cm = - std::sqrt( TF( 1 ) / 9 );
    const TF sm = + std::sqrt( TF( 8 ) / 9 );
    const TF qm = + std::sqrt( TF( 2 ) / 3 );

    const TI n0 = add_node( englobing_center + 4 * englobing_radius * Pt{  1,    0,        0 } );
    const TI n1 = add_node( englobing_center + 4 * englobing_radius * Pt{ cm,    0,     - sm } );
    const TI n2 = add_node( englobing_center + 4 * englobing_radius * Pt{ cm, + qm, 0.5 * sm } );
    const TI n3 = add_node( englobing_center + 4 * englobing_radius * Pt{ cm, - qm, 0.5 * sm } );

    const TI e0 = add_straight_edge( n0, n1, 0 );
    const TI e1 = add_straight_edge( n1, n2, 0 );
    const TI e2 = add_straight_edge( n2, n0, 0 );
    const TI e3 = add_straight_edge( n0, n3, 0 );
    const TI e4 = add_straight_edge( n3, n1, 0 );
    const TI e5 = add_straight_edge( n2, n3, 0 );

    auto add_face = [&]( TI e0, TI e1, TI e2 ) {
        const Pt &P0 = node_pos( edges[ e0 ].n0 );
        const Pt &P1 = node_pos( edges[ e0 ].n1 );
        const Pt &P2 = node_pos( edges[ e1 ].n1 );
        const Pt  n  = normalized( cross_prod( P0 - P1, P2 - P1 ) );
        const TI  ci = add_cut_info( node_pos( edges[ e0 ].n0 ), n, englobing_cut_id );
        add_flat_surface( add_edge_indices( e0, e1, e2 ), ci );
    };
    add_face( e2 + 1, e1 + 1, e0 + 1 );
    add_face( e0 + 0, e4 + 1, e3 + 1 );
    add_face( e5 + 0, e4 + 0, e1 + 0 );
    add_face( e2 + 0, e3 + 0, e5 + 1 );
}


template<class Pc>
typename ConvexPolyhedron3<Pc>::Pt ConvexPolyhedron3<Pc>::node_pos( TI num_node ) const {
    return { pos[ num_node + 0 * nb_glued_nodes ], pos[ num_node + 1 * nb_glued_nodes ], pos[ num_node + 2 * nb_glued_nodes ] };
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::node_sp( TI num_node ) const {
    return pos[ num_node + 3 * nb_glued_nodes ];
}

template<class Pc>
void ConvexPolyhedron3<Pc>::clear( const Box &box, CI cut_id ) {
    round_surfaces.resize( 0 );
    flat_surfaces      .resize( 0 );
    edge_indices       .clear();
    cut_info           .resize( 0 );
    edges              .resize( 0 );
    holes              .resize( 0 );
    pos                .resize( 4 * 2 * nb_glued_nodes );
    rpos               = pos.size() / 4;
    _nb_nodes          = 0;
    sphere_radius      = -1;

    const TI n0 = add_node( Pt{ box.p0.x, box.p0.y, box.p0.z } );
    const TI n1 = add_node( Pt{ box.p1.x, box.p0.y, box.p0.z } );
    const TI n2 = add_node( Pt{ box.p0.x, box.p1.y, box.p0.z } );
    const TI n3 = add_node( Pt{ box.p1.x, box.p1.y, box.p0.z } );
    const TI n4 = add_node( Pt{ box.p0.x, box.p0.y, box.p1.z } );
    const TI n5 = add_node( Pt{ box.p1.x, box.p0.y, box.p1.z } );
    const TI n6 = add_node( Pt{ box.p0.x, box.p1.y, box.p1.z } );
    const TI n7 = add_node( Pt{ box.p1.x, box.p1.y, box.p1.z } );

    const TI e0 = add_straight_edge( n0, n1, 0 );
    const TI e1 = add_straight_edge( n1, n3, 0 );
    const TI e2 = add_straight_edge( n3, n2, 0 );
    const TI e3 = add_straight_edge( n2, n0, 0 );

    const TI e4 = add_straight_edge( n4, n6, 0 );
    const TI e5 = add_straight_edge( n6, n7, 0 );
    const TI e6 = add_straight_edge( n7, n5, 0 );
    const TI e7 = add_straight_edge( n5, n4, 0 );

    const TI e8 = add_straight_edge( n0, n4, 0 );
    const TI e9 = add_straight_edge( n1, n5, 0 );
    const TI ea = add_straight_edge( n3, n7, 0 );
    const TI eb = add_straight_edge( n2, n6, 0 );

    auto add_face = [&]( TI e0, TI e1, TI e2, TI e3 ) {
        const Pt &P0 = node_pos( edges[ e0 ].n0 );
        const Pt &P1 = node_pos( edges[ e0 ].n1 );
        const Pt &P2 = node_pos( edges[ e1 ].n1 );
        const Pt  n  = normalized( cross_prod( P0 - P1, P2 - P1 ) );
        const TI  ci = add_cut_info( P0, n, cut_id );
        add_flat_surface( add_edge_indices( e0, e1, e2, e3 ), ci );
    };

    add_face( e0 + 0, e1 + 0, e2 + 0, e3 + 0 );
    add_face( e4 + 0, e5 + 0, e6 + 0, e7 + 0 );

    add_face( e8 + 0, e7 + 1, e9 + 1, e0 + 1 );
    add_face( ea + 0, e5 + 1, eb + 1, e2 + 1 );

    add_face( eb + 0, e4 + 1, e8 + 1, e3 + 1 );
    add_face( e9 + 0, e6 + 1, ea + 1, e1 + 1 );
}

template<class Pc>
void ConvexPolyhedron3<Pc>::add_centroid_contrib( FunctionEnum::Unit, Pt &ctd, TF &mea ) const {
    // base
    if ( flat_surfaces.empty() ) {
        TF vol = 4 * M_PI / 3 * std::pow( std::max( TF( 0 ), sphere_radius ), 3 );
        ctd += vol * sphere_center;
        mea += vol;
    } else {
        // center of the pyramids
        Pt sc = sphere_radius < 0 ? cut_info[ flat_surfaces[ 0 ].cut_index ].cut_O : sphere_center;

        // round surfaces
        if ( ! round_surfaces.empty() ) {
            Pt s_ctd;
            TF s_mea;
            _get_centroid_rf( s_ctd, s_mea );
            TF vol = sphere_radius * s_mea / 3;
            ctd += vol * ( sc + TF( 3 ) / 4 * ( s_ctd / s_mea - sc ) );
            mea += vol;
        }

        // flat surfaces
        for( const FlatSurface &fp : flat_surfaces ) {
            TF sgd = dot( cut_info[ fp.cut_index ].cut_O - sc, cut_info[ fp.cut_index ].cut_N );
            Pt s_ctd;
            TF s_mea;
            _get_centroid( s_ctd, s_mea, fp );
            if ( s_mea ) {
                TF vol = sgd * s_mea / 3;
                ctd += vol * ( sc + TF( 3 ) / 4 * ( s_ctd / s_mea - sc ) );
                mea += vol;
            }
        }
    }

    // holes
    for( const Hole &hole : holes ) {
        TF h = sphere_radius - dot( cut_info[ hole.cut_index ].cut_O - sphere_center, cut_info[ hole.cut_index ].cut_N );
        TF c = 3 * std::pow( 2 * sphere_radius - h, 2 ) / ( 4 * ( 3 * sphere_radius - h ) );
        TF v = M_PI / 3 * h * h * ( 3 * sphere_radius - h );
        ctd -= v * ( sphere_center + c * cut_info[ hole.cut_index ].cut_N );
        mea -= v;
    }
}

template<class Pc>
bool ConvexPolyhedron3<Pc>::contains( const Pt &pos ) const {
    for( const CutInfo &ci : cut_info )
        if ( dot( pos - ci.cut_O, ci.cut_N ) >= 0 )
            return false;
    return true;
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::Pt ConvexPolyhedron3<Pc>::centroid( FunctionEnum::Unit ) const {
    Pt ctd = { 0, 0, 0 };
    TF mea = 0;

    add_centroid_contrib( ctd, mea );

    return mea ? ctd / mea : ctd;
}

template<class Pc>

typename Pc::TF ConvexPolyhedron3<Pc>::measure( FunctionEnum::Unit ) const {
    TF res;
    if ( flat_surfaces.empty() ) {
        res = 4 * M_PI / 3 * std::pow( std::max( TF( 0 ), sphere_radius ), 3 );
    } else if ( round_surfaces.empty() ) {
        res = 0;
        for( const FlatSurface &fp : flat_surfaces )
            res += dot( cut_info[ fp.cut_index ].cut_O, cut_info[ fp.cut_index ].cut_N ) * area( fp ) / 3;
    } else {
        if ( round_surfaces.size() == 1 ) {
            res = sphere_radius * area( round_surfaces[ 0 ] ) / 3;
        } else {
            // we substract area of the hole from area of the full sphere
            TF sa = 4 * M_PI * std::pow( sphere_radius, 2 );
            TF pa = sa * ( TF( 1 ) - nb_connections );
            for( const RoundSurface &rp : round_surfaces )
                pa += area( rp );
            res = sphere_radius * pa / 3;
        }

        const Pt &sc = sphere_center;
        for( const FlatSurface &fp : flat_surfaces )
            res += dot( cut_info[ fp.cut_index ].cut_O - sc, cut_info[ fp.cut_index ].cut_N ) * area( fp ) / 3;
    }

    for( const Hole &hole : holes ) {
        TF h = sphere_radius - dot( cut_info[ hole.cut_index ].cut_O - sphere_center, cut_info[ hole.cut_index ].cut_N );
        res -= M_PI / 3 * h * h * ( 3 * sphere_radius - h );
    }
    return res;
}

template<class Pc>
typename Pc::TF ConvexPolyhedron3<Pc>::boundary_measure( FunctionEnum::Unit ) const {
    TF res;
    if ( flat_surfaces.empty() ) {
        res = 4 * M_PI * std::pow( std::max( TF( 0 ), sphere_radius ), 2 );
    } else if ( round_surfaces.empty() ) {
        res = 0;
        for( const FlatSurface &fp : flat_surfaces )
            res += area( fp );
    } else {
        if ( round_surfaces.size() == 1 ) {
            res = area( round_surfaces[ 0 ] );
        } else {
            // we substract area of the hole from area of the full sphere
            TF sa = 4 * M_PI * std::pow( sphere_radius, 2 );
            res = sa * ( TF( 1 ) - nb_connections );
            for( const RoundSurface &rp : round_surfaces )
                res += area( rp );
        }

        for( const FlatSurface &fp : flat_surfaces )
            res += area( fp );
    }

    for( const Hole &hole : holes ) {
        TF s = dot( cut_info[ hole.cut_index ].cut_O - sphere_center, cut_info[ hole.cut_index ].cut_N );
        TF r = std::sqrt( sphere_radius * sphere_radius - s * s );
        res += M_PI * ( r * r - 2 * sphere_radius * ( sphere_radius - s ) );
    }
    return res;
}

template<class Pc> template<class Fu>
typename ConvexPolyhedron3<Pc>::Pt ConvexPolyhedron3<Pc>::centroid_ap( const Fu &fu, TI n ) const {
    auto rdm1 = []() {
        return 2.0 * rand() / ( RAND_MAX - 1.0 ) - 1.0;
    };

    auto inside_semi_planes = [&]( Pt p ) {
        for( const CutInfo &ci : cut_info )
            if ( dot( p - ci.cut_O, ci.cut_N ) >= 0 )
                return false;
        return true;
    };

    TF count{ 0 };
    Pt centroid{ 0, 0, 0 };
    if ( sphere_radius < 0 ) {
        Pt mi = { + std::numeric_limits<TF>::max(), + std::numeric_limits<TF>::max(), + std::numeric_limits<TF>::max() };
        Pt ma = - mi;
        for( std::size_t n = 0; n < nb_points(); ++n ) {
            mi = min( mi, point( n ) );
            ma = max( ma, point( n ) );
        }
        for( TI i = 0; i < n; ++i ) {
            Pt p{ mi.x + ( ma.x - mi.x ) * rdm1(), mi.y + ( ma.y - mi.y ) * rdm1(), mi.z + ( ma.z - mi.z ) * rdm1() };
            if ( inside_semi_planes( p ) ) {
                centroid += fu( p, sphere_center ) * p;
                count += fu( p, sphere_center );
            }
        }
    } else {
        for( TI i = 0; i < n; ++i ) {
            Pt p{ rdm1(), rdm1(), rdm1() };
            if ( norm_2_p2( p ) <= 1 && inside_semi_planes( sphere_center + sphere_radius * p ) ) {
                centroid += fu( p, sphere_center ) * ( sphere_center + sphere_radius * p );
                count += fu( p, sphere_center );
            }
        }
    }

    return centroid / ( count + ( count == 0 ) );
}

template<class Pc> template<class Fu>
typename Pc::TF ConvexPolyhedron3<Pc>::measure_ap( const Fu &fu, TI n ) const {
    // width of the random distribution
    Pt sc = sphere_center;
    TF sr = sphere_radius;
    if ( sphere_radius < 0 ) {
        sc = { 0, 0, 0 };
        for( std::size_t n = 0; n < nb_points(); ++n )
            sc += point( n );
        sc /= TF( nb_points() );

        sr = 0;
        for( std::size_t n = 0; n < nb_points(); ++n )
            sr = std::max( sr, norm_2( point( n ) - sc ) );
    }

    auto rdm1 = []() {
        return 2.0 * rand() / ( RAND_MAX - 1.0 ) - 1.0;
    };

    auto inside_semi_planes = [&]( Pt p ) {
        for( const CutInfo &ci : cut_info )
            if ( dot( p - ci.cut_O, ci.cut_N ) >= 0 )
                return false;
        return true;
    };

    TF count = 0;
    for( TI i = 0; i < n; ++i ) {
        Pt p{ rdm1(), rdm1(), rdm1() };
        count += norm_2_p2( p ) <= 1 && inside_semi_planes( sc + sr * p );
    }

    return count * std::pow( 2 * sr, 3 ) / n;
}

template<class Pc> template<class Fu>
typename Pc::TF ConvexPolyhedron3<Pc>::boundary_measure_ap( const Fu &fu, TF max_ratio_area_error ) const {
    AreaOutput<Fu,TF> ao;
    display( ao, 0, true, max_ratio_area_error );
    return ao.area;
}

template<class Pc>

void ConvexPolyhedron3<Pc>::add_round_surface(const std::vector<TI> &edges ) {
    round_surfaces.push_back( { edges } );
}

template<class Pc>

void ConvexPolyhedron3<Pc>::add_flat_surface( std::pair<TI,TI> edge_indices_bounds, TI cut_index ) {
    flat_surfaces.push_back( { edge_indices_bounds.first, edge_indices_bounds.second, cut_index } );
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TI ConvexPolyhedron3<Pc>::add_straight_edge( TI n0, TI n1, TI cut_index ) {
    Edge edge;
    edge.n0        = n0;
    edge.n1        = n1;
    edge.radius    = -1;
    edge.cut_index = cut_index;

    // positive version
    TI res = edges.size();
    edges.push_back( edge );

    // negative one
    std::swap( n0, n1 );
    edge.n0        = n0;
    edge.n1        = n1;
    edges.push_back( edge );

    // return index on positive version
    return res;
}

template<class Pc>
std::pair<typename ConvexPolyhedron3<Pc>::TI,typename ConvexPolyhedron3<Pc>::TI> ConvexPolyhedron3<Pc>::add_edge_indices( TI e0, TI e1, TI e2, TI e3 ) {
    TI beg = edge_indices.size();
    edge_indices.push_back( e0 );
    edge_indices.push_back( e1 );
    edge_indices.push_back( e2 );
    edge_indices.push_back( e3 );
    return { beg, edge_indices.size() };
}

template<class Pc>
std::pair<typename ConvexPolyhedron3<Pc>::TI,typename ConvexPolyhedron3<Pc>::TI> ConvexPolyhedron3<Pc>::add_edge_indices( TI e0, TI e1, TI e2 ) {
    TI beg = edge_indices.size();
    edge_indices.push_back( e0 );
    edge_indices.push_back( e1 );
    edge_indices.push_back( e2 );
    return { beg, edge_indices.size() };
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TI ConvexPolyhedron3<Pc>::add_round_edge( TI n0, TI n1, TI cut_index ) {
    Edge edge;
    edge.n0        = n0;
    edge.n1        = n1;
    edge.cut_index = cut_index;
    edge.center    = sphere_center + dot( node_pos( n0 ) - sphere_center, cut_info[ cut_index ].cut_N ) * cut_info[ cut_index ].cut_N;
    edge.radius    = norm_2( node_pos( n0 ) - edge.center );

    // TODO: handle small edge radii
    if ( edge.radius ) {
        edge.X         = normalized( node_pos( n0 ) - edge.center );
        edge.tangent_0 = cross_prod( edge.X, cut_info[ cut_index ].cut_N );
        edge.tangent_1 = normalized( cross_prod( node_pos( n1 ) - edge.center, cut_info[ cut_index ].cut_N ) );
        edge.angle_1   = angle( edge, node_pos( n1 ) );
    } else {
        edge.X         = ortho_rand( cut_info[ cut_index ].cut_N );
        edge.tangent_0 = cross_prod( edge.X, cut_info[ cut_index ].cut_N );
        edge.tangent_1 = edge.tangent_0;
        edge.angle_1   = 0;
    }

    // positive version
    TI res = edges.size();
    edges.push_back( edge );

    // negative one
    std::swap( n0, n1 );
    edge.n0        = n0;
    edge.n1        = n1;
    // TODO: handle small edge radii
    if ( edge.radius ) {
        edge.X         = normalized( node_pos( n0 ) - edge.center );
        edge.tangent_0 = - cross_prod( edge.X, cut_info[ cut_index ].cut_N );
        edge.tangent_1 = - normalized( cross_prod( node_pos( n1 ) - edge.center, cut_info[ cut_index ].cut_N ) );
        edge.angle_1   = angle( edge, node_pos( n1 ) );
    } else {
        edge.X         = ortho_rand( cut_info[ cut_index ].cut_N );
        edge.tangent_0 = cross_prod( edge.X, cut_info[ cut_index ].cut_N );
        edge.tangent_1 = edge.tangent_0;
        edge.angle_1   = 0;
    }
    edges.push_back( edge );

    // return index on positive version
    return res;
}

template<class Pc>
typename Pc::TF ConvexPolyhedron3<Pc>::angle( const Edge &edge, Pt p ) const {
    return atan2p( dot( p - edge.center, edge.Y() ), dot( p - edge.center, edge.X ) );
}

template<class Pc>
typename Pc::TF ConvexPolyhedron3<Pc>::area( const RoundSurface &rp ) const {
    TF res = 2 * M_PI;

    auto kg = [&]( const Edge &edge ) {
        TF s = dot( node_pos( edge.n0 ) - sphere_center, cut_info[ edge.cut_index ].cut_N ) > 0 ? -1 : +1;
        return s * std::sqrt( std::max( TF( 0 ), std::pow( sphere_radius, 2 ) - std::pow( edge.radius, 2 ) ) ) / sphere_radius * edge.angle_1;
    };

    for( TI i = rp.end_in_edge_indices - 1, j = rp.beg_in_edge_indices; j < rp.end_in_edge_indices; i = j++ )
        res -= std::acos( std::max( std::min( dot( edges[ edge_indices[ i ] ].tangent_1, edges[ edge_indices[ j ] ].tangent_0 ), TF( 1 ) ), TF( -1 ) ) );
    for( TI n = rp.beg_in_edge_indices; n < rp.end_in_edge_indices; ++n )
        res -= kg( edges[ edge_indices[ n ] ] );

    return std::pow( sphere_radius, 2 ) * res;
}

template<class Pc>
typename Pc::TF ConvexPolyhedron3<Pc>::area( const FlatSurface &fs ) const {
    // area of straight triangles
    TF poly_area = 0;
    const Pt &normal = cut_info[ fs.cut_index ].cut_N;
    const Edge &e0 = edges[ edge_indices[ fs.beg_in_edge_indices ] ];
    for( TI num_in_edge_indices = fs.beg_in_edge_indices + 1; num_in_edge_indices < fs.end_in_edge_indices; ++num_in_edge_indices ) {
        const Edge &e1 = edges[ edge_indices[ num_in_edge_indices ] ];
        poly_area += dot( cross_prod(
            node_pos( e1.n1 ) - node_pos( e0.n0 ),
            node_pos( e1.n0 ) - node_pos( e0.n0 )
        ), normal );
    }
    poly_area /= 2;

    // area of circular caps
    TF caps_area = 0;
    for( TI num_in_edge_indices = fs.beg_in_edge_indices; num_in_edge_indices < fs.end_in_edge_indices; ++num_in_edge_indices ) {
        const Edge &edge = edges[ edge_indices[ num_in_edge_indices ] ];
        if ( edge.round() ) {
            const Pt &Pi = node_pos( edge.n0 );
            const Pt &Pj = node_pos( edge.n1 );

            TF Ri = edge.radius;
            TF d0 = norm_2( Pj - Pi ) / 2;
            TF d1 = sqrt( std::max( TF( 0 ), std::pow( Ri, 2 ) - std::pow( d0, 2 ) ) );
            TF a1 = edge.angle_1;

            if ( a1 < M_PI )
                caps_area -= d0 * d1;
            else
                caps_area += d0 * d1;

            caps_area += a1 / 2 * std::pow( Ri, 2 );
        }
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
void ConvexPolyhedron3<Pc>::_get_centroid_rf( Pt &centroid, TF &area ) const {
    centroid = { 0, 0, 0 };
    area = 0;
    for_each_triangle_rf( [&]( Pt P0, Pt P1, Pt P2 ) {
        TF a = norm_2( cross_prod( P1 - P0, P2 - P0 ) ) / 2;
        centroid += a / 3 * ( P0 + P1 + P2 );
        area += a;
    }, 1e-2, false );
}

template<class Pc>
void ConvexPolyhedron3<Pc>::_get_centroid( Pt &centroid, TF &area, const FlatSurface &fs ) const {
    centroid = { 0, 0, 0 };
    area = 0;

    // straight triangles
    const Pt &normal = cut_info[ fs.cut_index ].cut_N;
    const Edge &e0 = edges[ edge_indices[ fs.beg_in_edge_indices ] ];
    for( TI i = fs.beg_in_edge_indices + 1; i < fs.end_in_edge_indices; ++i ) {
        const Edge &e1 = edges[ edge_indices[ i ] ];
        Pt P0 = node_pos( e0.n0 );
        Pt P1 = node_pos( e1.n0 );
        Pt P2 = node_pos( e1.n1 );

        TF a = dot( cross_prod( P2 - P0, P1 - P0 ), normal ) / 2;
        centroid += a / 3 * ( P0 + P1 + P2 );
        area += a;
    }

    // circular caps
    for( TI i = fs.beg_in_edge_indices; i < fs.end_in_edge_indices; ++i ) {
        const Edge &edge = edges[ edge_indices[ i ] ];
        if ( edge.round() ) {
            const Pt &Pi = node_pos( edge.n0 );
            const Pt &Pj = node_pos( edge.n1 );

            TF Ri = edge.radius;
            TF d0 = norm_2( Pj - Pi ) / 2;
            TF d1 = sqrt( std::max( TF( 0 ), std::pow( Ri, 2 ) - std::pow( d0, 2 ) ) );
            TF a1 = edge.angle_1;
            TF ta = d1 * d0; // triangle area
            Pt tc = ta / 3 * ( edge.center + Pi + Pj ); // triangle centroid * area
            TF pa = a1 / 2 * std::pow( Ri, 2 ); // pie area
            Pt pc = pa * edge.center + 2.0 / 3.0 * std::pow( edge.radius, 3 ) * std::sin( a1 / 2 ) *
                    ( std::cos( a1 / 2 ) * edge.X + std::sin( a1 / 2 ) * edge.Y() ); // pie centroid * area

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

template<class Pc>
void ConvexPolyhedron3<Pc>::remove_unused_nodes( TI old_nodes_size ) {
    using std::max;

    // old nodes (which may have been removed)
    TI node_index = 0;
    new_node_indices.resize( max( old_nodes_size, nb_points() ) );
    for( TI n = 0; n < old_nodes_size; ++n ) {
        if ( ! sp( n ) < 0 ) {
            while ( ++n < old_nodes_size ) {
                if ( sp( n ) < 0 ) {
                    set_point( node_index, point( n ) );
                    new_node_indices[ n ] = node_index++;
                }
            }
            break;
        }
        new_node_indices[ n ] = node_index++;
    }

    // new nodes (which are inside by construction)
    for( TI n = old_nodes_size; n < nb_points(); ++n ) {
        set_point( node_index, point( n ) );
        new_node_indices[ n ] = node_index++;
    }

    // -> update node indices
    for( Edge &edge : edges ) {
        edge.n0 = new_node_indices[ edge.n0 ];
        edge.n1 = new_node_indices[ edge.n1 ];
    }

    // new size
    _nb_nodes = node_index;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::remove_unused_edges( TI old_edges_size, bool new_edges_are_round ) {
    // mark used edges
    for( TI n = 0; n < old_edges_size; ++n )
        edges[ n ].nedge = 0;
    for( TI ind : edge_indices ) {
        ind &= ~ TI( 1 );
        if ( ind < old_edges_size ) {
            edges[ ind + 0 ].nedge = 1;
            edges[ ind + 1 ].nedge = 1;
        }
    }

    // old edges (which may not be used)
    TI edge_index = 0;
    for( TI n = 0; n < old_edges_size; ++n ) {
        if ( edges[ n ].nedge == 0 ) {
            while( ++n < old_edges_size ) {
                if ( edges[ n ].nedge ) {
                    edges[ edge_index ].cut_index = edges[ n ].cut_index;
                    edges[ edge_index ].n0        = edges[ n ].n0       ;
                    edges[ edge_index ].n1        = edges[ n ].n1       ;
                    edges[ n ].nedge = edge_index++;
                }
            }
            break;
        }
        edges[ n ].nedge = edge_index++;
    }

    // new edges (which will necessarily be used)
    if ( new_edges_are_round ) {
        for( TI n = old_edges_size; n < edges.size(); ++n ) {
            edges[ edge_index ].cut_index = edges[ n ].cut_index;
            edges[ edge_index ].n0        = edges[ n ].n0       ;
            edges[ edge_index ].n1        = edges[ n ].n1       ;
            edges[ edge_index ].tangent_0 = edges[ n ].tangent_0;
            edges[ edge_index ].tangent_1 = edges[ n ].tangent_1;
            edges[ edge_index ].angle_1   = edges[ n ].angle_1  ;
            edges[ edge_index ].center    = edges[ n ].center   ;
            edges[ edge_index ].radius    = edges[ n ].radius   ;
            edges[ edge_index ].X         = edges[ n ].X        ;

            edges[ n ].nedge = edge_index++;
        }
    } else {
        for( TI n = old_edges_size; n < edges.size(); ++n ) {
            edges[ edge_index ].cut_index = edges[ n ].cut_index;
            edges[ edge_index ].n0        = edges[ n ].n0       ;
            edges[ edge_index ].n1        = edges[ n ].n1       ;

            edges[ n ].nedge = edge_index++;
        }
    }

    // update edge indices
    for( TI &ind : edge_indices )
        ind = edges[ ind ].nedge;

    // update edges.size
    edges.resize( edge_index );
}

template<class Pc>
void ConvexPolyhedron3<Pc>::remove_unused_cuts() {
    // mark used cuts
    for( CutInfo &ci : cut_info )
        ci.used = 0;
    for( TI n = 0; n < edges.size(); n += 2 )
        cut_info[ edges[ n ].cut_index ].used = 1;
    for( const FlatSurface &rs : flat_surfaces )
        cut_info[ rs.cut_index ].used = 1;
    for( const Hole &hole : holes )
        cut_info[ hole.cut_index ].used = 1;

    //
    TI cut_index = 0;
    for( TI n = 0; n < cut_info.size(); ++n ) {
        if ( cut_info[ n ].used == 0 ) {
            while( ++n < cut_info.size() ) {
                if ( cut_info[ n ].used ) {
                    cut_info[ cut_index ].cut_id = std::move( cut_info[ n ].cut_id );
                    cut_info[ cut_index ].cut_N  = std::move( cut_info[ n ].cut_N  );
                    cut_info[ cut_index ].cut_O  = std::move( cut_info[ n ].cut_O  );
                    cut_info[ n ].used = cut_index++;
                }
            }
            break;
        }
        cut_info[ n ].used = cut_index++;
    }

    // update indices
    for( Edge &edge : edges )
        edge.cut_index = cut_info[ edge.cut_index ].used;
    for( FlatSurface &rs : flat_surfaces )
        rs.cut_index = cut_info[ rs.cut_index ].used;
    for( Hole &hole : holes )
        hole.cut_index = cut_info[ hole.cut_index ].used;

    // resize
    cut_info.resize( cut_index );
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TI ConvexPolyhedron3<Pc>::_make_edge_cut( std::vector<Pt> &pts, std::map<std::pair<TI, ConvexPolyhedron3::TI>, TI> &edge_cuts, TI P0, TI P1, Pt PT ) {
    if ( P0 > P1 )
        std::swap( P0, P1 );
    auto edid = std::make_pair( P0, P1 );
    auto iter = edge_cuts.find( edid );
    if ( iter != edge_cuts.end() ) {
        std::size_t res = iter->second;
        edge_cuts.erase( iter );
        return res;
    }
    TI res = pts.size();
    edge_cuts.emplace_hint( iter, std::make_pair( edid, res ) );
    pts.emplace_back( PT );
    return res;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::_get_connected_points( std::vector<bool> &connected, const std::vector<std::vector<ConvexPolyhedron3::TI> > &connected_points, ConvexPolyhedron3::TI index ) {
    if ( connected[ index ] == false ) {
        connected[ index ] = true;

        for( TI new_index : connected_points[ index ] )
            _get_connected_points( connected, connected_points, new_index );
    }
}

template<class Pc>
void ConvexPolyhedron3<Pc>::_get_connections_rec( TI num_connection, TI num_node ) {
    if ( num_connections[ num_node ] == num_connection )
        return;
    num_connections[ num_node ] = num_connection;

    for( TI i = ( num_node ? new_node_indices[ num_node - 1 ] : 0 ); i < new_node_indices[ num_node ]; ++i )
        _get_connections_rec( num_connection, node_connectivity[ i ] );
}

template<class Pc>
void ConvexPolyhedron3<Pc>::get_edge_points( std::vector<Pt> &points, const Edge &edge, int nb_divs, bool end ) const {
    if ( edge.straight() ) {
        points.push_back( node_pos( edge.n0 ) );
        if ( end )
            points.push_back( node_pos( edge.n1 ) );
    } else {
        for( int j = 0, n = std::ceil( edge.angle_1 * nb_divs / ( 2 * M_PI ) ); j < n + end; ++j )
            points.push_back( point_for_angle( edge, edge.angle_1 * j / n ) );
    }
}

template<class Pc> template<class V>
void ConvexPolyhedron3<Pc>::display( V &vo, const typename V::CV &cell_data, bool filled, TF max_ratio_area_error, bool display_tangents ) const {
    vo.mutex.lock();

    // full or empty sphere ?
    if ( filled ) {

        // round surfaces
        for_each_triangle_rf( [&]( Pt p0, Pt p1, Pt p2 ) {
            vo.add_polygon( { p0, p1, p2 }, cell_data );
        }, max_ratio_area_error, true );

        // flat surfaces
        for( const FlatSurface &fs : flat_surfaces ) {
            std::vector<Pt> points;
            for( TI num_edge_indices = fs.beg_in_edge_indices; num_edge_indices < fs.end_in_edge_indices; ++num_edge_indices )
                get_edge_points( points, edges[ edge_indices[ num_edge_indices ] ], max_ratio_area_error > 2e-2 ? 50 : 500 );
            vo.add_polygon( points, cell_data );
            //            for( std::size_t i = 2; i < points.size(); ++i )
            //                vo.add_polygon( { points[ 0 ], points[ i - 1 ], points[ i - 0 ] }, cell_data );
        }

        // hole planes
        for( const Hole &hole : holes ) {
            TF s = dot( cut_info[ hole.cut_index ].cut_O - sphere_center, cut_info[ hole.cut_index ].cut_N );
            Pt O = sphere_center + s * cut_info[ hole.cut_index ].cut_N;
            Pt X = ortho_rand( cut_info[ hole.cut_index ].cut_N );
            Pt Y = cross_prod( cut_info[ hole.cut_index ].cut_N, X );
            TF r = std::sqrt( sphere_radius * sphere_radius - s * s );
            std::vector<Pt> points;
            for( TI i = 0, d = max_ratio_area_error > 2e-2 ? 50 : 500; i < d; ++i )
                points.push_back( O + r * std::cos( i * 2 * M_PI / d ) * X + r * std::sin( i * 2 * M_PI / d ) * Y );
            vo.add_polygon( points, cell_data );
        }
    } else {
        for( TI i = 0; i < edges.size() / 2; ++i ) {
            const Edge &edge = edges[ 2 * i ];
            std::vector<Pt> points;
            get_edge_points( points, edge, 50, true );
            vo.add_lines( points, cell_data );
        }
    }

    if ( display_tangents ) {
        for( TI i = 0; i < edges.size() / 2; ++i ) {
            const Edge &edge = edges[ 2 * i ];
            vo.add_arrow( node_pos( edge.n0 ), edge.tangent_0, cell_data );
            vo.add_arrow( node_pos( edge.n1 ), edge.tangent_1, cell_data );
        }
    }

    vo.mutex.unlock();
}

template<class Pc> template<class F>
void ConvexPolyhedron3<Pc>::for_each_triangle_rf( F &&func, TF max_ratio_area_error, bool remove_holes, std::mutex *m ) const {
    if ( sphere_radius <= 0 )
        return;

    // types
    using Triangle = typename SubdividedIcosahedron<TF>::Triangle;
    using Mesh     = typename SubdividedIcosahedron<TF>::Mesh;

    // start with a full mesh of the sphere
    static SubdividedIcosahedron<TF> si;
    static std::mutex m_si;
    m_si.lock();
    const Mesh &mesh = si.mesh_for_error( max_ratio_area_error );
    m_si.unlock();
    std::vector<Triangle> triangles = mesh.triangles;
    std::vector<Pt> points( mesh.points.size() );
    for( size_t i = 0; i < mesh.points.size(); ++i )
        points[ i ] = sphere_center + sphere_radius * mesh.points[ i ];

    // cut with edges
    if ( remove_holes ) {
        for( const CutInfo &ci : cut_info )
            p_cut( triangles, points, ci.cut_O, ci.cut_N );
    } else {
        for( const FlatSurface &fs : flat_surfaces )
            p_cut( triangles, points, cut_info[ fs.cut_index ].cut_O, cut_info[ fs.cut_index ].cut_N );
    }

    // display
    if ( m ) m->lock();
    for( const Triangle &triangle : triangles )
        func( points[ triangle.P0 ], points[ triangle.P1 ], points[ triangle.P2 ] );
    if ( m ) m->unlock();
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
