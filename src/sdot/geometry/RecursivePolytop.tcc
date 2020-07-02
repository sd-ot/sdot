#include "../support/for_each_comb.h"
#include "../support/ASSERT.h"
#include "../support/P.h"
#include "RecursivePolytop.h"

template<class TF,int dim,class TI,class UserNodeData>
RecursivePolytop<TF,dim,TI,UserNodeData>::RecursivePolytop( std::initializer_list<Pt> pts ) : RecursivePolytop( pts.size() ) {
    TI num = 0;
    for( const Pt &pos : pts ) {
        vertices[ num ].pos = pos;
        vertices[ num ].num = num;
        ++num;
    }
}

template<class TF,int dim,class TI,class UserNodeData>
RecursivePolytop<TF,dim,TI,UserNodeData>::RecursivePolytop( TI nb_vertices ) : vertices{ pool, nb_vertices }, date( 0 ) {
}

template<class TF,int dim,class TI,class UserNodeData>
void RecursivePolytop<TF,dim,TI,UserNodeData>::make_convex_hull() {
    // center
    impl.center = TF( 0 );
    for( TI i = 0; i < vertices.size(); ++i )
        impl.center += vertices[ i ].pos;
    impl.center /= TF( vertices.size() );

    // allowed indices
    std::vector<TI> indices( dim * ( vertices.size() + dim ) );
    for( TI i = 0; i < vertices.size(); ++i )
        indices[ i ] = i;

    //
    std::array<Pt,dim> prev_normals, prev_dirs;
    impl.add_convex_hull( pool, vertices.data(), indices.data(), vertices.size(), prev_normals.data(), prev_dirs.data() );
}

template<class TF,int dim,class TI,class UserNodeData>
void RecursivePolytop<TF,dim,TI,UserNodeData>::write_to_stream( std::ostream &os, std::string nl, std::string ns ) const {
    for( const Vertex &v : vertices )
        os << v.pos << " " << v.num << "; ";
    impl.for_each_item_rec( [&]( const auto &face ) {
        os << nl;
        for( TI i = 0; i < dim - face.nvi; ++i )
            os << ns;
        face.write_to_stream( os );
    } );
}

template<class TF,int dim,class TI,class UserNodeData> template<class VO>
void RecursivePolytop<TF,dim,TI,UserNodeData>::display_vtk( VO &vo ) const {
    // normals
    impl.for_each_item_rec( [&]( const auto &face ) {
        typename VO::Pt O = 0, N = 0;
        for( TI d = 0; d < std::min( int( dim ), 3 ); ++d ) {
            O[ d ] = conv( face.center[ d ], S<typename VO::TF>() );
            N[ d ] = conv( face.normal[ d ], S<typename VO::TF>() );
        }
        if ( norm_2( N ) )
            N /= norm_2( N );
        vo.add_line( { O, O + N } );
    } );

    // edges
    impl.for_each_item_rec( [&]( const auto &edge ) {
        std::vector<typename VO::Pt> pts;
        for( const auto *v : edge.vertices )
            pts.push_back( v->pos );

        vo.add_line( pts );
    }, N<1>() );

    // faces
    impl.for_each_item_rec( [&]( const auto &face ) {
        for( auto &edge : face.faces )
            edge.vertices[ 0 ]->tmp_v = edge.vertices[ 1 ];

        std::vector<typename VO::Pt> pts;
        for( const Vertex *b = face.faces.first().vertices[ 0 ], *v = b; ; v = v->tmp_v ) {
            pts.push_back( v->pos );
            if ( v->tmp_v == b )
                break;
        }

        vo.add_polygon( pts );
    }, N<2>() );
}

template<class TF,int dim,class TI,class UserNodeData>
TF RecursivePolytop<TF,dim,TI,UserNodeData>::measure() const {
    std::array<Pt,dim> dirs;
    return impl.measure( dirs ) / factorial( TF( int( dim ) ) );
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

    // get nb interpolated vertices
    std::vector<bool> cr_edge( vertices.size() * ( vertices.size() - 1 ), false );
    impl.for_each_item_rec( [&]( const auto &edge ) {
        Vertex *v = edge.vertices[ 0 ], *w = edge.vertices[ 1 ];
        if ( bool( v->tmp_f > 0 ) != bool( w->tmp_f > 0 ) ) {
            TI n0 = min( v->num, w->num );
            TI n1 = max( v->num, w->num );
            TI nn = n1 * ( n1 - 1 ) + n0;
            if ( ! cr_edge[ nn ] ) {
                cr_edge[ nn ] = true;
                ++new_vertices_size;
            }
        }
    }, N<1>() );

    // copy of inside vertices
    RecursivePolytop<TF,dim,TI,UserNodeData> res( new_vertices_size );

    new_vertices_size = 0;
    for( const Vertex &v : vertices ) {
        if ( v.tmp_f <= 0 ) {
            Vertex &nv = res.vertices[ new_vertices_size ];
            nv.user_data = v.user_data;
            nv.pos = v.pos;

            nv.num = new_vertices_size++;
        }
    }

    // make the interpolated vertices
    std::vector<Vertex *> new_vertices( vertices.size() * ( vertices.size() - 1 ), nullptr );
    impl.for_each_item_rec( [&]( const auto &edge ) {
        Vertex *v = edge.vertices[ 0 ], *w = edge.vertices[ 1 ];
        if ( bool( v->tmp_f > 0 ) != bool( w->tmp_f > 0 ) ) {
            TI n0 = min( v->num, w->num );
            TI n1 = max( v->num, w->num );
            TI nn = n1 * ( n1 - 1 ) + n0;
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

    //
    IntrusiveList<typename Impl::Face> new_faces;
    impl.plane_cut( res.impl, res.pool, new_faces, new_vertices, date );
    return res;
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


