#include "../support/for_each_comb.h"
#include "../support/ASSERT.h"
#include "../support/P.h"
#include "RecursivePolytop.h"

template<class TF,int dim,class TI,class UserNodeData>
RecursivePolytop<TF,dim,TI,UserNodeData>::RecursivePolytop() : date( 0 ) {
}

template<class TF,int dim,class TI,class UserNodeData>
RecursivePolytop<TF,dim,TI,UserNodeData> RecursivePolytop<TF,dim,TI,UserNodeData>::convex_hull( const std::vector<Node> &nodes ) {
    RecursivePolytop res;

    // make the convex hull recursively
    std::vector<TI> indices( dim * ( nodes.size() + dim ) );
    for( TI i = 0; i < nodes.size(); ++i )
        indices[ i ] = i;
    std::array<Pt,dim> prev_normals;
    res.impl.add_convex_hull( res.pool, nodes.data(), indices.data(), nodes.size(), prev_normals.data(), res.date, N<dim>() );

    // adjust orientation and node ordering
    //    std::array<Pt,dim> dirs;
    //    res.impl.sort_vertices( dirs, N<dim>() );

    return res;
}

template<class TF,int dim,class TI,class UserNodeData>
void RecursivePolytop<TF,dim,TI,UserNodeData>::write_to_stream( std::ostream &os, std::string nl, std::string ns ) const {
    impl.for_each_item_rec( [&]( const auto &face ) {
        os << nl;
        for( TI i = 0; i < dim - face.nvi; ++i )
            os << ns;
        face.write_to_stream( os );
    } );
}

template<class TF,int dim,class TI,class UserNodeData> template<class VO>
void RecursivePolytop<TF,dim,TI,UserNodeData>::display_vtk( VO &vo ) const {
    impl.for_each_item_rec( [&]( const auto &face ) {
        typename VO::Pt O = 0, N = 0;
        for( TI d = 0; d < std::min( int( dim ), 3 ); ++d ) {
            O[ d ] = conv( face.center[ d ], S<typename VO::TF>() );
            N[ d ] = conv( face.normal[ d ], S<typename VO::TF>() );
        }
        if ( norm_2( N ) )
            N /= norm_2( N );
        vo.add_line( { O, O + N } );

        //        std::vector<typename VO::Pt> pts( face.vertices.size() );
        //        for( TI i = 0; i < pts.size(); ++i )
        //            for( TI d = 0; d < std::min( int( dim ), 3 ); ++d )
        //                pts[ i ][ d ] = conv( face.vertices[ i ]->node.pos[ d ], S<typename VO::TF>() );

        //        if ( face.nvi == 1 )
        //            vo.add_line( pts );
        //        if ( face.nvi == 2 )
        //            vo.add_polygon( pts );
    } );
}

template<class TF,int dim,class TI,class UserNodeData>
TF RecursivePolytop<TF,dim,TI,UserNodeData>::measure() const {
    std::array<Pt,dim> dirs;
    return impl.measure( dirs, N<dim>() ) / factorial( TF( int( dim ) ) );
}

template<class TF,int dim,class TI,class UserNodeData>
void RecursivePolytop<TF,dim,TI,UserNodeData>::plane_cut( Pt orig, Pt normal, const std::function<UserNodeData(const UserNodeData &,const UserNodeData &,TF,TF)> &nf ) {
    using std::min;
    using std::max;

    //    // scalar product + index for each vertex
    //    TI nb_vertices = 0;
    //    for( Vertex *v : impl.vertices ) {
    //        v->tmp_f = dot( v->node.pos - orig, normal );
    //        v->ind = nb_vertices++;
    //    }

    //    // make the interpolated vertices
    //    std::vector<Vertex *> new_vertices( nb_vertices * ( nb_vertices - 1 ), nullptr );
    //    impl.for_each_item_rec( [&]( const auto &face ) {
    //        if ( face.nvi == 1 && face.vertices.size() == 2 ) {
    //            Vertex *v = face.vertices[ 0 ], *w = face.vertices[ 1 ];
    //            if ( bool( v->tmp_f > 0 ) != bool( w->tmp_f > 0 ) ) {
    //                TI n0 = min( v->ind, w->ind );
    //                TI n1 = max( v->ind, w->ind );
    //                TI nn = n1 * ( n1 - 1 ) + n0;
    //                if ( ! new_vertices[ nn ] ) {
    //                    Vertex *nv = pool.template create<Vertex>();
    //                    new_vertices[ nn ] = nv;

    //                    nv->node.pos = v->node.pos + v->tmp_f / ( v->tmp_f - w->tmp_f ) * ( w->node.pos - v->node.pos );
    //                    if ( nf )
    //                        nv->node.user_data = nf( v->node.user_data, w->node.user_data, v->tmp_f, w->tmp_f );
    //                }
    //            }
    //        }
    //    } );

    //    //
    //    IntrusiveList<typename Impl::Face> new_faces;
    //    impl.plane_cut( pool, new_faces, new_vertices, date, N<dim>() );
    //    P( new_faces );
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


