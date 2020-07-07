#include "../support/for_each_permutation.h"
#include "../support/for_each_comb.h"
#include "../support/ASSERT.h"
#include "../support/range.h"
#include "../support/P.h"
#include "RecursivePolytop.h"

template<class TF,int dim,class TI,class UserNodeData>
RecursivePolytop<TF,dim,TI,UserNodeData>::RecursivePolytop( std::initializer_list<Pt> pts ) : RecursivePolytop( pts.size() ) {
    TI num = 0;
    for( const Pt &pos : pts )
        vertices[ num++ ].pos = pos;
}

template<class TF,int dim,class TI,class UserNodeData>
RecursivePolytop<TF,dim,TI,UserNodeData>::RecursivePolytop( const std::vector<Pt> &pts ) : RecursivePolytop( pts.size() ) {
    for( TI num = 0; num < pts.size(); ++num )
        vertices[ num ].pos = pts[ num ];
}


template<class TF,int dim,class TI,class UserNodeData>
RecursivePolytop<TF,dim,TI,UserNodeData>::RecursivePolytop( TI nb_vertices ) : date( 0 ), vertices{ pool, nb_vertices } {
    for( TI num = 0; num < nb_vertices; ++num )
        vertices[ num ].num = num;
}

template<class TF,int dim,class TI,class UserNodeData>
TI RecursivePolytop<TF,dim,TI,UserNodeData>::nb_faces() const {
    TI res = 0;
    for( const Impl &impl : impls )
        res += impl.faces.size();
    return res;
}

template<class TF,int dim,class TI,class UserNodeData>
bool RecursivePolytop<TF,dim,TI,UserNodeData>::valid_vertex_prop( const std::vector<Pt> &pts ) const {
    for( const Impl &impl : impls )
        if ( ! impl.valid_vertex_prop( pts ) )
            return false;
    return true;
}

template<class TF,int dim,class TI,class UserNodeData>
RecursivePolytop<TF,dim,TI,UserNodeData> RecursivePolytop<TF,dim,TI,UserNodeData>::with_points( const std::vector<Pt> &pts ) const {
    RecursivePolytop<TF,dim,TI,UserNodeData> res( vertices.size() );
    for( TI i = 0; i < vertices.size(); ++i ) {
        res.vertices[ i ].user_data = vertices[ i ].user_data;
        res.vertices[ i ].pos = pts[ vertices[ i ].num ];
    }

    for( const Impl &impl : impls )
        impl.with_points( res.impls, res.pool, res.vertices.data() );

    res.update_normals();

    return res;
}

template<class TF,int dim,class TI,class UserNodeData> template<class F,int n>
void RecursivePolytop<TF,dim,TI,UserNodeData>::for_each_item_rec( const F &fu, N<n> ) const {
    for( const Impl &impl : impls )
        impl.for_each_item_rec( fu, N<n>() );
}

template<class TF,int dim,class TI,class UserNodeData> template<class F>
void RecursivePolytop<TF,dim,TI,UserNodeData>::for_each_item_rec( const F &fu ) const {
    for( const Impl &impl : impls )
        impl.for_each_item_rec( fu );
}

template<class TF,int dim,class TI,class UserNodeData>
void RecursivePolytop<TF,dim,TI,UserNodeData>::make_convex_hull() {
    // allowed indices
    std::vector<TI> indices( dim * ( vertices.size() + dim ) );
    for( TI i = 0; i < vertices.size(); ++i )
        indices[ i ] = i;

    // add the faces
    std::array<Pt,dim> prev_normals, prev_dirs;
    Pt center = mean( vertices, Vertex::get_pos );
    Impl::add_convex_hull( impls, *this, indices.data(), vertices.size(), prev_normals.data(), prev_dirs.data(), Pt( TF( 0 ) ), center );
}

template<class TF,int dim,class TI,class UserNodeData>
void RecursivePolytop<TF,dim,TI,UserNodeData>::write_to_stream( std::ostream &os, std::string nl, std::string ns ) const {
    for( const Vertex &v : vertices )
        os << v.pos << " " << v.num << "; ";
    for( const Impl &impl : impls ) {
        impl.for_each_item_rec( [&]( const auto &face ) {
            os << nl;
            for( TI i = 0; i < dim - face.nvi; ++i )
                os << ns;
            face.write_to_stream( os );
        } );
    }
}

template<class TF,int dim,class TI,class UserNodeData>
void RecursivePolytop<TF,dim,TI,UserNodeData>::update_normals() {
    Pt center = mean( vertices, Vertex::get_pos );
    std::array<Pt,dim> prev_normals;
    std::array<TI,dim> indices;
    for( Impl &impl : impls )
        for( typename Impl::Face &face : impl.faces )
            face.update_normals( prev_normals.data(), vertices.data(), indices.data(), center );
}

template<class TF,int dim,class TI,class UserNodeData>
bool RecursivePolytop<TF,dim,TI,UserNodeData>::contains( const Pt &pt ) const {
    for( const Impl &impl : impls )
        if ( impl.contains( pt ) )
            return true;
    return false;
}

template<class TF,int dim,class TI,class UserNodeData>
bool RecursivePolytop<TF,dim,TI,UserNodeData>::all_vertices_are_used() const {
    for( const Vertex &v : vertices )
        v.tmp_f = 0;

    for( Impl &impl : impls ) {
        impl.for_each_vertex( [&]( const Vertex *v ) {
            v->tmp_f = 1;
        } );
    }

    for( const Vertex &v : vertices )
        if ( v.tmp_f == 0 )
            return false;
    return true;
}

template<class TF,int dim,class TI,class UserNodeData>
bool RecursivePolytop<TF,dim,TI,UserNodeData>::can_use_perm_pts( const Pt *pts, TI *num_in_pts, bool want_convexity ) const {
    return ! for_each_permutation_cont<TI>( range<TI>( vertices.size() ), [&]( const std::vector<TI> &prop_num_in_pts ) {
        //
        std::vector<Pt> npts( vertices.size() );
        for( TI i = 0; i < vertices.size(); ++i )
            npts[ i ] = pts[ prop_num_in_pts[ i ] ];
        RecursivePolytop rp = with_points( npts );
        if ( rp.measure() <= 0 )
            return true;
        if ( want_convexity && ! rp.is_convex() )
            return true;

        for( TI i = 0; i < vertices.size(); ++i )
            num_in_pts[ i ] = prop_num_in_pts[ i ];
        return false;
    } );
}

template<class TF,int dim,class TI,class UserNodeData>
bool RecursivePolytop<TF,dim,TI,UserNodeData>::is_convex() const {
    for( Impl &impl : impls ) {
        for( typename Impl::Face &face : impl.faces ) {
            Pt orig = face.first_vertex()->pos;
            for( const Vertex &vertex : vertices )
                if ( dot( vertex.pos - orig, face.normal ) > 0 )
                    return false;
        }
    }
    return true;
}

template<class TF,int dim,class TI,class UserNodeData> template<class VO>
void RecursivePolytop<TF,dim,TI,UserNodeData>::display_vtk( VO &vo ) const {
    using std::min;

    // normals
    for( const Impl &impl : impls ) {
        impl.for_each_item_rec( [&]( const auto &face ) {
            Pt C = face.center();
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
            for( const auto *v : edge.vertices ) {
                typename VO::Pt pt;
                for( TI d = 0; d < min( int( dim ), 3 ); ++d )
                    pt[ d ] = conv( v->pos[ d ], S<typename VO::TF>() );
                pts.push_back( pt );
            }

            vo.add_line( pts );
        }, N<1>() );
    }

    // faces
    for( const Impl &impl : impls ) {
        impl.for_each_item_rec( [&]( const auto &face ) {
            for( const Vertex &vertex : vertices )
                vertex.next = nullptr;
            for( auto &edge : face.faces )
                edge.vertices[ 0 ]->next = edge.vertices[ 1 ];

            std::vector<typename VO::Pt> pts;
            for( const Vertex *b = face.faces.first().vertices[ 0 ], *v = b; ; v = v->next ) {
                typename VO::Pt pt;
                for( TI d = 0; d < min( int( dim ), 3 ); ++d )
                    pt[ d ] = conv( v->pos[ d ], S<typename VO::TF>() );
                pts.push_back( pt );
                if ( v->next == b || ! v->next )
                    break;
            }

            vo.add_polygon( pts );
        }, N<2>() );
    }
}

template<class TF,int dim,class TI,class UserNodeData>
TF RecursivePolytop<TF,dim,TI,UserNodeData>::measure() const {
    TF res = 0;
    std::array<Pt,dim> dirs;
    for( const Impl &impl : impls )
        res += impl.measure( dirs, impl.first_vertex()->pos );
    return res / factorial( TF( int( dim ) ) );
}

template<class TF,int dim,class TI,class UserNodeData>
RecursivePolytop<TF,dim,TI,UserNodeData> RecursivePolytop<TF,dim,TI,UserNodeData>::plane_cut( Pt orig, Pt normal, const std::function<UserNodeData(const UserNodeData &,const UserNodeData &,TF,TF)> &nf ) const {
    using std::min;
    using std::max;

    // scalar product for each vertex + copy of vertices that are inside
    TI new_vertices_size = 0;
    for( const Vertex &v : vertices ) {
        v.tmp_f = dot( v.pos - orig, normal );
        new_vertices_size += v.tmp_f <= 0;
    }

    // get the number of interpolated vertices to create
    std::vector<bool> cr_edge( vertices.size() * ( vertices.size() - 1 ) / 2, false );
    for( const Impl &impl : impls ) {
        impl.for_each_item_rec( [&]( const auto &edge ) {
            Vertex *v = edge.vertices[ 0 ], *w = edge.vertices[ 1 ];
            if ( bool( v->tmp_f > 0 ) != bool( w->tmp_f > 0 ) ) {
                TI n0 = min( v->num, w->num );
                TI n1 = max( v->num, w->num );
                TI nn = n1 * ( n1 - 1 ) / 2 + n0;
                if ( ! cr_edge[ nn ] ) {
                    cr_edge[ nn ] = true;
                    ++new_vertices_size;
                }
            }
        }, N<1>() );
    }

    // Rp to return
    RecursivePolytop<TF,dim,TI,UserNodeData> res( new_vertices_size );

    // copy of inside vertices
    new_vertices_size = 0;
    for( const Vertex &v : vertices ) {
        if ( v.tmp_f <= 0 ) {
            Vertex &nv = res.vertices[ new_vertices_size ];
            nv.user_data = v.user_data;
            nv.pos = v.pos;

            nv.num = new_vertices_size++;
            v.tmp_v = &nv;
        }
    }

    // make the interpolated vertices
    std::vector<Vertex *> new_vertices( vertices.size() * ( vertices.size() - 1 ) / 2, nullptr );
    for( const Impl &impl : impls ) {
        impl.for_each_item_rec( [&]( const auto &edge ) {
            Vertex *v = edge.vertices[ 0 ], *w = edge.vertices[ 1 ];
            if ( bool( v->tmp_f > 0 ) != bool( w->tmp_f > 0 ) ) {
                TI n0 = min( v->num, w->num );
                TI n1 = max( v->num, w->num );
                TI nn = n1 * ( n1 - 1 ) / 2 + n0;
                if ( ! new_vertices[ nn ] ) {
                    Vertex &nv = res.vertices[ new_vertices_size ];
                    new_vertices[ nn ] = &nv;

                    nv.pos = v->pos + v->tmp_f / ( v->tmp_f - w->tmp_f ) * ( w->pos - v->pos );
                    if ( nf )
                        nv.user_data = nf( v->user_data, w->user_data, v->tmp_f, w->tmp_f );

                    nv.num = new_vertices_size++;
                }
            }
        }, N<1>() );
    }

    //
    IntrusiveList<typename Impl::Face> io_faces;
    const Vertex *io_vertex;
    for( const Impl &impl : impls )
        impl.plane_cut( res.impls, res, *this, new_vertices, io_faces, io_vertex, normal, N<dim>() );
    return res;
}

template<class TF,int dim,class TI,class UserNodeData> template<class Rpi>
void RecursivePolytop<TF,dim,TI,UserNodeData>::make_tmp_connections( const IntrusiveList<Rpi> &impls ) const {
    using std::min;
    using std::max;

    // reset vertex.beg that will be used to get nb connected vertices per vertex
    for( const Vertex &vertex : vertices )
        vertex.beg = 0;

    // nb connected vertices
    tmp_edges.assign( vertices.size() * ( vertices.size() - 1 ) / 2, false );
    for( const auto &impl : impls ) {
        impl.for_each_item_rec( [&]( const auto &edge ) {
            Vertex *v = edge.vertices[ 0 ], *w = edge.vertices[ 1 ];
            TI n0 = min( v->num, w->num );
            TI n1 = max( v->num, w->num );
            TI nn = n1 * ( n1 - 1 ) / 2 + n0;
            if ( ! tmp_edges[ nn ] ) {
                tmp_edges[ nn ] = true;
                ++v->beg;
                ++w->beg;
            }
        }, N<1>() );
    }

    // scan
    TI tmp_connections_size = 0;
    for( const Vertex &vertex : vertices ) {
        TI old_tmp_connections_size = tmp_connections_size;
        tmp_connections_size += vertex.beg;

        vertex.beg = old_tmp_connections_size;
        vertex.end = old_tmp_connections_size;
    }

    //
    tmp_edges.assign( vertices.size() * ( vertices.size() - 1 ) / 2, false );
    tmp_connections.resize( tmp_connections_size );
    for( const auto &impl : impls ) {
        impl.for_each_item_rec( [&]( const auto &edge ) {
            Vertex *v = edge.vertices[ 0 ], *w = edge.vertices[ 1 ];
            TI n0 = min( v->num, w->num );
            TI n1 = max( v->num, w->num );
            TI nn = n1 * ( n1 - 1 ) / 2 + n0;
            if ( ! tmp_edges[ nn ] ) {
                tmp_edges[ nn ] = true;
                tmp_connections[ v->end++ ] = w;
                tmp_connections[ w->end++ ] = v;
            }
        }, N<1>() );
    }
}

template<class TF,int dim,class TI,class UserNodeData> template<class Rpi>
TI RecursivePolytop<TF,dim,TI,UserNodeData>::get_connected_graphs( const IntrusiveList<Rpi> &items ) const {
    make_tmp_connections( items );

    // find connected node
    TI res = 0, ori_date = ++date;
    while( true ) {
        // find a node that has connection and that has not been seen so far
        const Vertex *c = nullptr;
        for( const Vertex &v : vertices ) {
            if ( v.beg != v.end && v.date < ori_date ) {
                c = &v;
                break;
            }
        }

        // if not found, it's done
        if ( c == nullptr )
            break;

        // else, mark recursively the connected vertices
        ++res;
        ++date;
        mark_connected_rec( c );
    }

    return res;
}

template<class TF,int dim,class TI,class UserNodeData>
void RecursivePolytop<TF,dim,TI,UserNodeData>::mark_connected_rec( const Vertex *v ) const {
    if ( v->date == date )
        return;
    v->date = date;

    for( TI ind = v->beg; ind < v->end; ++ind )
        mark_connected_rec( tmp_connections[ ind ] );
}

template<class TF,int dim,class TI,class UserNodeData>
TI RecursivePolytop<TF,dim,TI,UserNodeData>::num_graph( const Vertex *v ) const {
    return date - v->date;
}

template<class TF,int dim,class TI,class UserNodeData>
TF RecursivePolytop<TF,dim,TI,UserNodeData>::measure_intersection( const Rp &a, const Rp &b ) {
    std::deque<std::array<Rp,2>> is;
    get_intersections( is, a, b );

    TF res = 0;
    for( std::array<Rp,2> &p : is ) {
        ASSERT( p[ 0 ].measure() == p[ 1 ].measure() );
        res += p[ 0 ].measure();
    }
    return res;
}

template<class TF,int dim,class TI,class UserNodeData>
void RecursivePolytop<TF,dim,TI,UserNodeData>::get_intersections( std::deque<std::array<Rp,2>> &res, const Rp &a, const Rp &b ) {
    bool first_time = true;
    auto for_each_prev_cuts = [&]( auto f ) {
        if ( first_time ) {
            first_time = false;
            f( a, b );
            return;
        }
        for( const std::array<Rp,2> &p : res )
            f( p[ 0 ], p[ 1 ] );
    };
    //
    for( const Rp *cutter : { &a, &b } ) {
        for( const Impl &impl : cutter->impls ) {
            for( const typename Impl::Face &face : impl.faces ) {
                std::deque<std::array<Rp,2>> tmp;
                for_each_prev_cuts( [&]( const Rp &a, const Rp &b ) {
                    for( Pt normal : { face.normal, - face.normal } ) {
                        Rp ca = a.plane_cut( face.first_vertex()->pos, normal );
                        if ( ca.measure() == 0 )
                            break;
                        Rp cb = b.plane_cut( face.first_vertex()->pos, normal );
                        if ( cb.measure() == 0 )
                            break;

                        tmp.push_back( {
                            std::move( ca ),
                            std::move( cb )
                        } );
                    }
                } );
                std::swap( tmp, res );
            }
        }
    }
}

//template<class TF,int nvi,int dim,class TI,class NodeData> template<class Nd>
//bool RecursivePolytop<TF,nvi,dim,TI,NodeData>::valid_node_prop( const std::vector<Nd> &prop, std::vector<Pt> prev_centers, bool prev_centers_are_valid ) const {
//    // available_points
//    std::vector<Pt> available_points;
//    available_points.reserve( prop.size() );
//    for( const Node &node : nodes )
//        if ( node.data < prop.size() )
//            available_points.push_back( prop[ node.data ].pos );

//    // check rank
//    TI r = rank( available_points );
//    if ( r > nvi )
//        return false;
//    if ( available_points.size() == nodes.size() && r != nvi )
//        return false;

//    // check faces
//    if ( r < nvi )
//        prev_centers_are_valid = false;
//    if ( prev_centers_are_valid )
//        prev_centers.push_back( mean( available_points ) );
//    for( const Face &face : faces )
//        if ( ! face.valid_node_prop( prop, prev_centers, prev_centers_are_valid ) )
//            return false;
//    return true;
//}

//template<class TF,int dim,class TI,class NodeData> template<class Nd>
//bool RecursivePolytop<TF,1,dim,TI,NodeData>::valid_node_prop( const std::vector<Nd> &prop, const std::vector<Pt> &prev_centers, bool prev_centers_are_valid ) const {
//    // available_points
//    std::vector<Pt> available_points;
//    available_points.reserve( prop.size() );
//    for( const Node &node : nodes )
//        if ( node.data < prop.size() )
//            available_points.push_back( prop[ node.data ].pos );

//    if ( available_points.size() == 2 ) {
//        if ( available_points[ 0 ] == available_points[ 1 ] )
//            return false;

//        if ( prev_centers_are_valid ) {
//            // check measure is > 0
//            std::vector<Pt> dirs;
//            for( TI i = 1; i < prev_centers.size(); ++i )
//                dirs.push_back( prev_centers[ i ] - prev_centers[ i - 1 ] );
//            if ( prev_centers.size() )
//                dirs.push_back( available_points[ 0 ] - prev_centers.back() );
//            dirs.push_back( available_points[ 1 ] - available_points[ 0 ] );
//            if ( determinant( &dirs[ 0 ][ 0 ], N<dim>() ) <= 0 )
//                return false;
//        }
//    }

//    return true;
//}
//std::vector<typename RecursivePolytop<TF,nvi,dim,TI,NodeData>::DN> RecursivePolytop<TF,nvi,dim,TI,NodeData>::non_closed_node_seq( const std::vector<Face> &faces ) {
//    std::deque<Node> pts;
//    std::set<TI> seen;
//    int dir;
//    auto start_new_seq = [&]() {
//        for( const Face &face : faces) {
//            for( const Node &node : face.nodes ) {
//                if ( ! seen.count( node.id ) ) {
//                    seen.insert( node.id );
//                    pts = { node };
//                    dir = 1;
//                    return true;
//                }
//            }
//        }
//        return false;
//    };

//    std::vector<std::deque<Node>> res;
//    if ( ! start_new_seq() )
//        return res;
//    while ( true ) {
//        for( TI num_edge = 0; ; ++num_edge ) {
//            if ( num_edge == faces.size() ) {
//                if ( dir > 0 ) {
//                    dir = 0;
//                    break;
//                }
//                res.push_back( pts );
//                if ( start_new_seq() )
//                    break;
//                return res;
//            }

//            const auto &edge = faces[ num_edge ];
//            if ( edge.nodes[ 1 - dir ].id == ( dir ? pts.back().id : pts.front().id ) ) {
//                if ( edge.nodes[ dir ].id == ( dir ? pts.front().id : pts.back().id ) ) { // in a loop ?
//                    if ( start_new_seq() )
//                        break;
//                    return res;
//                }
//                if ( dir )
//                    pts.push_back( edge.nodes[ 1 ] );
//                else
//                    pts.push_front( edge.nodes[ 0 ] );
//                seen.insert( edge.nodes[ dir ].id );
//                break;
//            }
//        }
//    }


//    return res;
//}


