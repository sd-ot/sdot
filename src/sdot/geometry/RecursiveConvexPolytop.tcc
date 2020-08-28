#include "RecursiveConvexPolytop.h"
#include "../support/ASSERT.h"

template<class TF,int dim,class TI>
RecursiveConvexPolytop<TF,dim,TI>::RecursiveConvexPolytop( const std::vector<Pt> &vertices ) : vertices( vertices ) {
    // allowed indices
    std::vector<TI> indices( dim * ( vertices.size() + dim ) );
    for( TI i = 0; i < vertices.size(); ++i )
        indices[ i ] = i;

    // add the faces
    std::array<Pt,dim> prev_normals, prev_dirs;
    Pt center = mean( vertices );
    Impl::add_convex_hull( impls, vertices.data(), indices.data(), vertices.size(), prev_normals.data(), prev_dirs.data(), Pt( TF( 0 ) ), center );
}

template<class TF,int dim,class TI>
void RecursiveConvexPolytop<TF,dim,TI>::write_to_stream( std::ostream &os, std::string nl, std::string ns ) const {
    for( const Pt &v : vertices )
        os << v << " ";
    for( const Impl &impl : impls ) {
        impl.for_each_item_rec( [&]( const auto &face ) {
            os << nl;
            for( TI i = 0; i < dim - face.nvi; ++i )
                os << ns;
            face.write_to_stream( os );
        } );
    }
}

template<class TF,int dim,class TI> template<class F,int n>
void RecursiveConvexPolytop<TF,dim,TI>::for_each_item_rec( const F &fu, N<n> ) const {
    for( const Impl &impl : impls )
        impl.for_each_item_rec( fu, N<n>() );
}

template<class TF,int dim,class TI> template<class F>
void RecursiveConvexPolytop<TF,dim,TI>::for_each_item_rec( const F &fu ) const {
    for( const Impl &impl : impls )
        impl.for_each_item_rec( fu );
}

template<class TF,int dim,class TI>
bool RecursiveConvexPolytop<TF,dim,TI>::contains( const Pt &pt ) const {
    for( const Impl &impl : impls )
        if ( impl.contains( vertices.data(), pt ) )
            return true;
    return false;
}

template<class TF,int dim,class TI>
RecursiveConvexPolytop<TF,dim,TI> RecursiveConvexPolytop<TF,dim,TI>::plane_cut( Pt orig, Pt normal ) const {
    using std::min;
    using std::max;

    std::vector<Pt> new_vertices;
    new_vertices.reserve( 2 * vertices.size() );

    // scalar product for each vertex + copy of vertices that are inside
    std::vector<TI> vertex_corr( vertices.size() );
    std::vector<TF> sp( vertices.size() );
    for( TI i = 0; i < vertices.size(); ++i ) {
        sp[ i ] = dot( vertices[ i ] - orig, normal );
        if ( ! ( sp[ i ] > 0 ) ) {
            vertex_corr[ i ] = new_vertices.size();
            new_vertices.push_back( vertices[ i ] );
        }
    }

    // get the number of interpolated vertices to create
    std::vector<TI> edge_corr( vertices.size() * ( vertices.size() - 1 ) / 2, 0 );
    for( const Impl &impl : impls ) {
        impl.for_each_item_rec( [&]( const auto &edge ) {
            TI v = edge.vertices[ 0 ], w = edge.vertices[ 1 ];
            if ( ( sp[ v ] > 0 ) != ( sp[ w ] > 0 ) ) {
                TI n0 = min( v, w ), n1 = max( v, w );
                TI nn = n1 * ( n1 - 1 ) / 2 + n0;
                if ( ! edge_corr[ nn ] ) {
                    edge_corr[ nn ] = new_vertices.size();
                    new_vertices.push_back( vertices[ v ] + sp[ v ] / ( sp[ v ] - sp[ w ] ) * ( vertices[ w ] - vertices[ v ] ) );
                }
            }
        }, N<1>() );
    }

    // make a convex hull from the new points
    return { new_vertices };
}

template<class TF,int dim,class TI> template<class VO>
void RecursiveConvexPolytop<TF,dim,TI>::display_vtk( VO &vo ) const {
    using std::min;

    // normals
    for( const Impl &impl : impls ) {
        impl.for_each_item_rec( [&]( const auto &face ) {
            Pt C = face.center( vertices.data() );
            typename VO::Pt O = 0, N = 0;
            for( TI d = 0; d < std::min( int( dim ), 3 ); ++d ) {
                N[ d ] = conv( face.normal[ d ], S<typename VO::TF>() );
                O[ d ] = conv( C[ d ], S<typename VO::TF>() );
            }
            if ( norm_2( N ) )
                N /= norm_2( N );
            vo.add_line( { O, O + N } );
        } );
    }

    // edges
    for( const Impl &impl : impls ) {
        impl.for_each_item_rec( [&]( const auto &edge ) {
            std::vector<typename VO::Pt> pts;
            for( TI v : edge.vertices ) {
                typename VO::Pt pt;
                for( TI d = 0; d < min( int( dim ), 3 ); ++d )
                    pt[ d ] = conv( vertices[ v ][ d ], S<typename VO::TF>() );
                pts.push_back( pt );
            }

            vo.add_line( pts );
        }, N<1>() );
    }

    // faces
    for( const Impl &impl : impls ) {
        impl.for_each_item_rec( [&]( const auto &face ) {
            std::vector<TI> nexts( vertices.size() );
            for( auto &edge : face.faces )
                nexts[ edge.vertices[ 0 ] ] = edge.vertices[ 1 ];

            std::vector<typename VO::Pt> pts;
            for( TI b = face.first_vertex(), v = b; ; v = nexts[ v ] ) {
                typename VO::Pt pt;
                for( TI d = 0; d < min( int( dim ), 3 ); ++d )
                    pt[ d ] = conv( vertices[ v ][ d ], S<typename VO::TF>() );
                pts.push_back( pt );
                if ( nexts[ v ] == b )
                    break;
            }

            vo.add_polygon( pts );
        }, N<2>() );
    }
}

template<class TF,int dim,class TI>
TF RecursiveConvexPolytop<TF,dim,TI>::measure() const {
    TF res = 0;
    std::array<Pt,dim> dirs;
    for( const Impl &impl : impls )
        res += impl.measure( vertices.data(), dirs, vertices[ impl.first_vertex() ] );
    return res / factorial( TF( int( dim ) ) );
}

//template<class TF,int dim,class TI> template<class Pts>
//auto RecursiveConvexPolytop<TF,dim,TI>::with_points( const Pts &pts ) const -> RecursiveConvexPolytop<typename std::decay<decltype( pts[ 0 ][ 0 ] )>::type,dim,TI,UserData> {
//    using NF = typename std::decay<decltype( pts[ 0 ][ 0 ] )>::type;
//    RecursiveConvexPolytop<NF,dim,TI> res( vertices.size() );
//    for( TI i = 0; i < vertices.size(); ++i ) {
//        res.vertices[ i ].user_data = vertices[ i ].user_data;
//        res.vertices[ i ].pos = pts[ vertices[ i ].num ];
//    }

//    for( const Impl &impl : impls )
//        impl.with_points( res.impls, res.pool, res.vertices.data() );

//    res.update_normals();

//    return res;
//}

//template<class TF,int dim,class TI>
//bool RecursiveConvexPolytop<TF,dim,TI>::all_vertices_are_used() const {
//    for( const Vertex &v : vertices )
//        v.tmp_f = 0;

//    for( Impl &impl : impls ) {
//        impl.for_each_vertex( [&]( const Vertex *v ) {
//            v->tmp_f = 1;
//        } );
//    }

//    for( const Vertex &v : vertices )
//        if ( v.tmp_f == 0 )
//            return false;
//    return true;
//}

//template<class TF,int dim,class TI>
//bool RecursiveConvexPolytop<TF,dim,TI>::can_use_perm_pts( const Pt *pts, TI *num_in_pts, bool want_convexity ) const {
//    return ! for_each_permutation_cont<TI>( range<TI>( vertices.size() ), [&]( const std::vector<TI> &prop_num_in_pts ) {
//        //
//        std::vector<Pt> npts( vertices.size() );
//        for( TI i = 0; i < vertices.size(); ++i )
//            npts[ i ] = pts[ prop_num_in_pts[ i ] ];
//        RecursiveConvexPolytop rp = with_points( npts );
//        if ( rp.measure() <= 0 )
//            return true;
//        if ( want_convexity && ! rp.is_convex() )
//            return true;

//        for( TI i = 0; i < vertices.size(); ++i )
//            num_in_pts[ i ] = prop_num_in_pts[ i ];
//        return false;
//    } );
//}

//template<class TF,int dim,class TI>
//bool RecursiveConvexPolytop<TF,dim,TI>::is_convex() const {
//    for( Impl &impl : impls ) {
//        for( typename Impl::Face &face : impl.faces ) {
//            Pt orig = face.first_vertex()->pos;
//            for( const Vertex &vertex : vertices )
//                if ( dot( vertex.pos - orig, face.normal ) > 0 )
//                    return false;
//        }
//    }
//    return true;
//}


//template<class TF,int dim,class TI> template<class Rpi>
//void RecursiveConvexPolytop<TF,dim,TI>::make_tmp_connections( const IntrusiveList<Rpi> &impls ) const {
//    using std::min;
//    using std::max;

//    // reset vertex.beg that will be used to get nb connected vertices per vertex
//    for( const Vertex &vertex : vertices )
//        vertex.beg = 0;

//    // nb connected vertices
//    tmp_edges.assign( vertices.size() * ( vertices.size() - 1 ) / 2, false );
//    for( const auto &impl : impls ) {
//        impl.for_each_item_rec( [&]( const auto &edge ) {
//            Vertex *v = edge.vertices[ 0 ], *w = edge.vertices[ 1 ];
//            TI n0 = min( v->num, w->num );
//            TI n1 = max( v->num, w->num );
//            TI nn = n1 * ( n1 - 1 ) / 2 + n0;
//            if ( ! tmp_edges[ nn ] ) {
//                tmp_edges[ nn ] = true;
//                ++v->beg;
//                ++w->beg;
//            }
//        }, N<1>() );
//    }

//    // scan
//    TI tmp_connections_size = 0;
//    for( const Vertex &vertex : vertices ) {
//        TI old_tmp_connections_size = tmp_connections_size;
//        tmp_connections_size += vertex.beg;

//        vertex.beg = old_tmp_connections_size;
//        vertex.end = old_tmp_connections_size;
//    }

//    //
//    tmp_edges.assign( vertices.size() * ( vertices.size() - 1 ) / 2, false );
//    tmp_connections.resize( tmp_connections_size );
//    for( const auto &impl : impls ) {
//        impl.for_each_item_rec( [&]( const auto &edge ) {
//            Vertex *v = edge.vertices[ 0 ], *w = edge.vertices[ 1 ];
//            TI n0 = min( v->num, w->num );
//            TI n1 = max( v->num, w->num );
//            TI nn = n1 * ( n1 - 1 ) / 2 + n0;
//            if ( ! tmp_edges[ nn ] ) {
//                tmp_edges[ nn ] = true;
//                tmp_connections[ v->end++ ] = w;
//                tmp_connections[ w->end++ ] = v;
//            }
//        }, N<1>() );
//    }
//}

//template<class TF,int dim,class TI> template<class Rpi>
//TI RecursiveConvexPolytop<TF,dim,TI>::get_connected_graphs( const IntrusiveList<Rpi> &items ) const {
//    make_tmp_connections( items );

//    // find connected node
//    TI res = 0, ori_date = ++date;
//    while( true ) {
//        // find a node that has connection and that has not been seen so far
//        const Vertex *c = nullptr;
//        for( const Vertex &v : vertices ) {
//            if ( v.beg != v.end && v.date < ori_date ) {
//                c = &v;
//                break;
//            }
//        }

//        // if not found, it's done
//        if ( c == nullptr )
//            break;

//        // else, mark recursively the connected vertices
//        ++res;
//        ++date;
//        mark_connected_rec( c );
//    }

//    return res;
//}

//template<class TF,int dim,class TI>
//void RecursiveConvexPolytop<TF,dim,TI>::mark_connected_rec( const Vertex *v ) const {
//    if ( v->date == date )
//        return;
//    v->date = date;

//    for( TI ind = v->beg; ind < v->end; ++ind )
//        mark_connected_rec( tmp_connections[ ind ] );
//}

//template<class TF,int dim,class TI>
//TI RecursiveConvexPolytop<TF,dim,TI>::num_graph( const Vertex *v ) const {
//    return date - v->date;
//}

//template<class TF,int dim,class TI>
//TF RecursiveConvexPolytop<TF,dim,TI>::measure_intersection( const Rp &a, const Rp &b ) {
//    std::deque<std::array<Rp,2>> is;
//    get_intersections( is, a, b );

//    TF res = 0;
//    for( std::array<Rp,2> &p : is ) {
//        ASSERT( p[ 0 ].measure() == p[ 1 ].measure() );
//        res += p[ 0 ].measure();
//    }
//    return res;
//}

//template<class TF,int dim,class TI>
//void RecursiveConvexPolytop<TF,dim,TI>::get_intersections( std::deque<std::array<Rp,2>> &res, const Rp &a, const Rp &b ) {
//    bool first_time = true;
//    auto for_each_prev_cuts = [&]( auto f ) {
//        if ( first_time ) {
//            first_time = false;
//            f( a, b );
//            return;
//        }
//        for( const std::array<Rp,2> &p : res )
//            f( p[ 0 ], p[ 1 ] );
//    };
//    //
//    for( const Rp *cutter : { &a, &b } ) {
//        for( const Impl &impl : cutter->impls ) {
//            for( const typename Impl::Face &face : impl.faces ) {
//                std::deque<std::array<Rp,2>> tmp;
//                for_each_prev_cuts( [&]( const Rp &a, const Rp &b ) {
//                    for( Pt normal : { face.normal, - face.normal } ) {
//                        Rp ca = a.plane_cut( face.first_vertex()->pos, normal );
//                        if ( ca.measure() == 0 )
//                            break;
//                        Rp cb = b.plane_cut( face.first_vertex()->pos, normal );
//                        if ( cb.measure() == 0 )
//                            break;

//                        tmp.push_back( {
//                            std::move( ca ),
//                            std::move( cb )
//                        } );
//                    }
//                } );
//                std::swap( tmp, res );
//            }
//        }
//    }
//}

////template<class TF,int nvi,int dim,class TI,class NodeData> template<class Nd>
////bool RecursiveConvexPolytop<TF,nvi,dim,TI,NodeData>::valid_node_prop( const std::vector<Nd> &prop, std::vector<Pt> prev_centers, bool prev_centers_are_valid ) const {
////    // available_points
////    std::vector<Pt> available_points;
////    available_points.reserve( prop.size() );
////    for( const Node &node : nodes )
////        if ( node.data < prop.size() )
////            available_points.push_back( prop[ node.data ].pos );

////    // check rank
////    TI r = rank( available_points );
////    if ( r > nvi )
////        return false;
////    if ( available_points.size() == nodes.size() && r != nvi )
////        return false;

////    // check faces
////    if ( r < nvi )
////        prev_centers_are_valid = false;
////    if ( prev_centers_are_valid )
////        prev_centers.push_back( mean( available_points ) );
////    for( const Face &face : faces )
////        if ( ! face.valid_node_prop( prop, prev_centers, prev_centers_are_valid ) )
////            return false;
////    return true;
////}

////template<class TF,int dim,class TI,class NodeData> template<class Nd>
////bool RecursiveConvexPolytop<TF,1,dim,TI,NodeData>::valid_node_prop( const std::vector<Nd> &prop, const std::vector<Pt> &prev_centers, bool prev_centers_are_valid ) const {
////    // available_points
////    std::vector<Pt> available_points;
////    available_points.reserve( prop.size() );
////    for( const Node &node : nodes )
////        if ( node.data < prop.size() )
////            available_points.push_back( prop[ node.data ].pos );

////    if ( available_points.size() == 2 ) {
////        if ( available_points[ 0 ] == available_points[ 1 ] )
////            return false;

////        if ( prev_centers_are_valid ) {
////            // check measure is > 0
////            std::vector<Pt> dirs;
////            for( TI i = 1; i < prev_centers.size(); ++i )
////                dirs.push_back( prev_centers[ i ] - prev_centers[ i - 1 ] );
////            if ( prev_centers.size() )
////                dirs.push_back( available_points[ 0 ] - prev_centers.back() );
////            dirs.push_back( available_points[ 1 ] - available_points[ 0 ] );
////            if ( determinant( &dirs[ 0 ][ 0 ], N<dim>() ) <= 0 )
////                return false;
////        }
////    }

////    return true;
////}
////std::vector<typename RecursiveConvexPolytop<TF,nvi,dim,TI,NodeData>::DN> RecursiveConvexPolytop<TF,nvi,dim,TI,NodeData>::non_closed_node_seq( const std::vector<Face> &faces ) {
////    std::deque<Node> pts;
////    std::set<TI> seen;
////    int dir;
////    auto start_new_seq = [&]() {
////        for( const Face &face : faces) {
////            for( const Node &node : face.nodes ) {
////                if ( ! seen.count( node.id ) ) {
////                    seen.insert( node.id );
////                    pts = { node };
////                    dir = 1;
////                    return true;
////                }
////            }
////        }
////        return false;
////    };

////    std::vector<std::deque<Node>> res;
////    if ( ! start_new_seq() )
////        return res;
////    while ( true ) {
////        for( TI num_edge = 0; ; ++num_edge ) {
////            if ( num_edge == faces.size() ) {
////                if ( dir > 0 ) {
////                    dir = 0;
////                    break;
////                }
////                res.push_back( pts );
////                if ( start_new_seq() )
////                    break;
////                return res;
////            }

////            const auto &edge = faces[ num_edge ];
////            if ( edge.nodes[ 1 - dir ].id == ( dir ? pts.back().id : pts.front().id ) ) {
////                if ( edge.nodes[ dir ].id == ( dir ? pts.front().id : pts.back().id ) ) { // in a loop ?
////                    if ( start_new_seq() )
////                        break;
////                    return res;
////                }
////                if ( dir )
////                    pts.push_back( edge.nodes[ 1 ] );
////                else
////                    pts.push_front( edge.nodes[ 0 ] );
////                seen.insert( edge.nodes[ dir ].id );
////                break;
////            }
////        }
////    }


////    return res;
////}


