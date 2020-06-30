#include "../support/for_each_comb.h"
#include "../support/ASSERT.h"
#include "../support/P.h"
#include "RecursivePolytop.h"
#include <algorithm>
#include <map>

template<class TF,int nvi,int dim,class TI,class NodeData>
RecursivePolytop<TF,nvi,dim,TI,NodeData> RecursivePolytop<TF,nvi,dim,TI,NodeData>::convex_hull( const std::vector<Node> &nodes, const std::vector<Pt> &prev_centers ) {
    struct DFace { std::array<Point<TF,nvi>,nvi> dirs; Point<TF,nvi> normal, orig; };
    std::vector<DFace> dfaces;
    RecursivePolytop res;

    // new center
    Pt center = TF( 0 );
    for( const Node &node : nodes )
        center += node.pos;
    center /= TF( nodes.size() );

    // => new center list
    std::vector<Pt> centers = prev_centers;
    centers.push_back( center );

    // new base. Return is base is not large enough
    std::vector<Pt> dirs;
    for( TI i = 1; i < nodes.size(); ++i )
        dirs.push_back( nodes[ i ].pos - nodes[ 0 ].pos );
    std::vector<Pt> base = base_from( dirs );
    ASSERT( base.size() <= nvi );
    if ( base.size() < nvi )
        return res;

    // local_pos (local coords)
    std::vector<Point<TF,nvi>> local_pos;
    for( const Node &node : nodes ) {
        std::array<std::array<TF,nvi>,nvi> M;
        std::array<TF,nvi> V;
        for( TI r = 0; r < nvi; ++r )
            for( TI c = 0; c < nvi; ++c )
                M[ r ][ c ] = dot( base[ r ], base[ c ] );
        for( TI r = 0; r < nvi; ++r )
            V[ r ] = dot( base[ r ], node.pos - center );

        std::array<TF,nvi> X = solve( M, V );
        local_pos.push_back( X.data() );
    }

    // try each way to make a face
    for_each_comb<TI>( nodes.size(), nvi, [&]( const std::vector<TI> &num_nodes ) {
        // make a face trial (to store it if it's new and relevant)
        DFace dface;
        dface.orig = local_pos[ num_nodes[ 0 ] ];
        for( TI d = 0; d < nvi - 1; ++d )
            dface.dirs[ d ] = local_pos[ num_nodes[ d + 1 ] ] - dface.orig;

        // get a normal
        dface.normal = cross_prod( dface.dirs.data() );
        if ( norm_2_p2( dface.normal ) == TF( 0 ) )
            return;

        // stop here if we already have this face
        for( const DFace &ecafd : dfaces )
            if ( dot( ecafd.normal, dface.orig - ecafd.orig ) == 0 && colinear( ecafd.normal, dface.normal ) )
                return;

        // stop here if it's not an "exterior" face
        bool has_ins = false;
        bool has_out = false;
        for( TI op = 0; op < nodes.size(); ++op ) {
            if ( op != num_nodes[ 0 ] ) {
                TF d = dot( local_pos[ op ] - dface.orig, dface.normal );
                has_ins |= d < 0;
                has_out |= d > 0;
            }
        }
        if ( has_ins && has_out )
            return;

        // update normal orientation if necessary
        if ( has_out )
            dface.normal *= TF( -1 );

        // register dface
        dfaces.push_back( dface );

        // find all the points that belong to this face
        std::vector<Node> new_nodes;
        for( TI op = 0; op < nodes.size(); ++op )
            if ( dot( local_pos[ op ] - dface.orig, dface.normal ) == TF( 0 ) )
                new_nodes.push_back( nodes[ op ] );

        // register nodes
        for( const Node &node : new_nodes )
            if ( std::find( res.nodes.begin(), res.nodes.end(), node ) == res.nodes.end() )
                res.nodes.push_back( node );

        // register face
        res.faces.push_back( Face::convex_hull( new_nodes, centers ) );
    } );

    // sort
    std::sort( res.nodes.begin(), res.nodes.end() );
    std::sort( res.faces.begin(), res.faces.end() );

    return res;
}

template<class TF,int dim,class TI,class NodeData>
RecursivePolytop<TF,1,dim,TI,NodeData> RecursivePolytop<TF,1,dim,TI,NodeData>::convex_hull( const std::vector<Node> &nodes, const std::vector<Pt> &prev_centers ) {
    ASSERT( nodes.size() >= 2 );

    // find dir
    Pt dir;
    for( TI i = 1; i < nodes.size(); ++i )
        if ( ( dir = nodes[ i ].pos - nodes[ 0 ].pos ) )
            break;

    // check orientation
    std::vector<Pt> dirs;
    for( TI i = 1; i < prev_centers.size(); ++i )
        dirs.push_back( prev_centers[ i ] - prev_centers[ i - 1 ] );
    if ( prev_centers.size() )
        dirs.push_back( nodes[ 0 ].pos - prev_centers.back() );
    dirs.push_back( dir );
    if ( determinant( &dirs[ 0 ][ 0 ], N<dim>() ) < 0 )
        dir *= TF( -1 );

    // get min and max points
    auto cmp_node = [&]( const Node &a, const Node &b ) { return dot( dir, a.pos ) < dot( dir, b.pos ); };
    TI n0 = std::distance( nodes.begin(), std::min_element( nodes.begin(), nodes.end(), cmp_node ) );
    TI n1 = std::distance( nodes.begin(), std::max_element( nodes.begin(), nodes.end(), cmp_node ) );
    return { { nodes[ n0 ], nodes[ n1 ] } };
}

template<class TF,int nvi,int dim,class TI,class NodeData>
void RecursivePolytop<TF,nvi,dim,TI,NodeData>::write_to_stream( std::ostream &os, std::string sp ) const {
    os << sp << "+ " << nodes;
    for( const Face &face : faces )
        face.write_to_stream( os, sp + "  " );
}

template<class TF,int dim,class TI,class NodeData>
void RecursivePolytop<TF,1,dim,TI,NodeData>::write_to_stream( std::ostream &os, std::string sp ) const {
    os << sp << "+ " << nodes;
}

template<class TF,int nvi,int dim,class TI,class NodeData>
bool RecursivePolytop<TF,nvi,dim,TI,NodeData>::operator<( const RecursivePolytop &that ) const {
    return faces < that.faces;
}

template<class TF,int dim,class TI,class NodeData>
bool RecursivePolytop<TF,1,dim,TI,NodeData>::operator<( const RecursivePolytop &that ) const {
    return nodes < that.nodes;
}

template<class TF,int nvi,int dim,class TI,class NodeData> template<class Fu>
void RecursivePolytop<TF,nvi,dim,TI,NodeData>::for_each_faces_rec( const Fu &func) const {
    func( *this );
    for( const Face &face : faces )
        face.for_each_faces_rec( func );
}

template<class TF,int dim,class TI,class NodeData> template<class Fu>
void RecursivePolytop<TF,1,dim,TI,NodeData>::for_each_faces_rec( const Fu &func ) const {
    func( *this );
}

template<class TF,int nvi,int dim,class TI,class NodeData> template<class Nd>
bool RecursivePolytop<TF,nvi,dim,TI,NodeData>::valid_node_prop( const std::vector<Nd> &prop, std::vector<Pt> prev_centers, bool prev_centers_are_valid ) const {
    // available_points
    std::vector<Pt> available_points;
    available_points.reserve( prop.size() );
    for( const Node &node : nodes )
        if ( node.data < prop.size() )
            available_points.push_back( prop[ node.data ].pos );

    // check rank
    TI r = rank( available_points );
    if ( r > nvi )
        return false;
    if ( available_points.size() == nodes.size() && r != nvi )
        return false;

    // check faces
    if ( r < nvi )
        prev_centers_are_valid = false;
    if ( prev_centers_are_valid )
        prev_centers.push_back( mean( available_points ) );
    for( const Face &face : faces )
        if ( ! face.valid_node_prop( prop, prev_centers, prev_centers_are_valid ) )
            return false;
    return true;
}

template<class TF,int dim,class TI,class NodeData> template<class Nd>
bool RecursivePolytop<TF,1,dim,TI,NodeData>::valid_node_prop( const std::vector<Nd> &prop, const std::vector<Pt> &prev_centers, bool prev_centers_are_valid ) const {
    // available_points
    std::vector<Pt> available_points;
    available_points.reserve( prop.size() );
    for( const Node &node : nodes )
        if ( node.data < prop.size() )
            available_points.push_back( prop[ node.data ].pos );

    if ( available_points.size() == 2 ) {
        if ( available_points[ 0 ] == available_points[ 1 ] )
            return false;

        if ( prev_centers_are_valid ) {
            // check measure is > 0
            std::vector<Pt> dirs;
            for( TI i = 1; i < prev_centers.size(); ++i )
                dirs.push_back( prev_centers[ i ] - prev_centers[ i - 1 ] );
            if ( prev_centers.size() )
                dirs.push_back( available_points[ 0 ] - prev_centers.back() );
            dirs.push_back( available_points[ 1 ] - available_points[ 0 ] );
            if ( determinant( &dirs[ 0 ][ 0 ], N<dim>() ) <= 0 )
                return false;
        }
    }

    return true;
}

template<class TF,int nvi,int dim,class TI,class NodeData>
std::vector<typename RecursivePolytop<TF,nvi,dim,TI,NodeData>::VN> RecursivePolytop<TF,nvi,dim,TI,NodeData>::node_seq() const {
    std::vector<Node> pts;
    std::set<TI> seen;
    auto start_new_seq = [&]() {
        for( const Node &node : nodes ) {
            if ( ! seen.count( node.id ) ) {
                seen.insert( node.id );
                pts = { node };
                return true;
            }
        }
        return false;
    };

    std::vector<std::vector<Node>> res;
    if ( ! start_new_seq() )
        return res;
    while ( true ) {
        for( TI num_edge = 0; ; ++num_edge ) {
            if ( num_edge == faces.size() ) {
                PE( "not a closed face" );
                res.push_back( pts );
                if ( start_new_seq() )
                    break;
                return res;
            }

            const auto &edge = faces[ num_edge ];
            if ( edge.nodes[ 0 ].id == pts.back().id ) {
                if ( edge.nodes[ 1 ].id == pts.front().id ) {
                    res.push_back( pts );
                    if ( start_new_seq() )
                        break;
                    return res;
                }
                pts.push_back( edge.nodes[ 1 ] );
                seen.insert( edge.nodes[ 1 ].id );
                break;
            }

        }
    }
}

template<class TF,int nvi,int dim,class TI,class NodeData> template<class Vk>
void RecursivePolytop<TF,nvi,dim,TI,NodeData>::display_vtk( Vk &vtk_output ) const {
    for_each_faces_rec( [&]( const auto &face ) {
        if constexpr ( face.nvi == 2 ) {
            for( const std::vector<Node> &ns : face.node_seq() ) {
                std::vector<typename Vk::Pt> pts( ns.size() );
                for( TI i = 0; i < ns.size(); ++i ) {
                    for( TI d = 0; d < std::min( dim, 3 ); ++d )
                        pts[ i ][ d ] = conv( ns[ i ].pos[ d ], S<typename Vk::TF>() );
                    for( TI d = std::min( dim, 3 ); d < 3; ++d )
                        pts[ i ][ d ] = 0;
                }
                vtk_output.add_polygon( pts );
            }
        }
    } );
}

template<class TF,int nvi,int dim,class TI,class NodeData> template<class Nd>
auto RecursivePolytop<TF,nvi,dim,TI,NodeData>::with_nodes( const std::vector<Nd> &new_nodes ) const {
    using NewNodeData = typename std::decay<decltype( new_nodes[ 0 ].data )>::type;
    RecursivePolytop<TF,nvi,dim,TI,NewNodeData> res;
    for( const Node &node : nodes )
        res.nodes.push_back( new_nodes[ node.data ] );
    for( const Face &face : faces )
        res.faces.push_back( face.with_nodes( new_nodes ) );
    return res;
}

template<class TF,int dim,class TI,class NodeData> template<class Nd>
auto RecursivePolytop<TF,1,dim,TI,NodeData>::with_nodes( const std::vector<Nd> &new_nodes ) const {
    using NewNodeData = typename std::decay<decltype( new_nodes[ 0 ].data )>::type;
    RecursivePolytop<TF,1,dim,TI,NewNodeData> res;
    for( const Node &node : nodes )
        res.nodes.push_back( new_nodes[ node.data ] );
    return res;
}

template<class TF,int nvi,int dim,class TI,class NodeData>
TI RecursivePolytop<TF,nvi,dim,TI,NodeData>::max_id() const {
    TI res = 0;
    for( const Node &node : nodes )
        res = std::max( res, node.id );
    return res;
}

template<class TF,int nvi,int dim,class TI,class NodeData>
RecursivePolytop<TF,nvi,dim,TI,NodeData> RecursivePolytop<TF,nvi,dim,TI,NodeData>::make_from( const std::vector<Face> &faces ) {
    RecursivePolytop res;
    res.faces = faces;
    for( const Face &face : faces )
        for( const Node &node : face.nodes )
            if ( std::find_if( res.nodes.begin(), res.nodes.end(), [&]( const Node &n ) { return n.id == node.id; } ) == res.nodes.end() )
                res.nodes.push_back( node );
    return res;
}

template<class TF,int dim,class TI,class NodeData>
RecursivePolytop<TF,1,dim,TI,NodeData> RecursivePolytop<TF,1,dim,TI,NodeData>::make_from( const std::vector<Face> &faces ) {
    RecursivePolytop res;
    res.nodes = faces;
    return res;
}

template<class TF,int nvi,int dim,class TI,class NodeData>
std::vector<typename RecursivePolytop<TF,nvi,dim,TI,NodeData>::DN> RecursivePolytop<TF,nvi,dim,TI,NodeData>::non_closed_node_seq( const std::vector<Face> &faces ) {
    std::deque<Node> pts;
    std::set<TI> seen;
    int dir;
    auto start_new_seq = [&]() {
        for( const Face &face : faces) {
            for( const Node &node : face.nodes ) {
                if ( ! seen.count( node.id ) ) {
                    seen.insert( node.id );
                    pts = { node };
                    dir = 1;
                    return true;
                }
            }
        }
        return false;
    };

    std::vector<std::deque<Node>> res;
    if ( ! start_new_seq() )
        return res;
    while ( true ) {
        for( TI num_edge = 0; ; ++num_edge ) {
            if ( num_edge == faces.size() ) {
                if ( dir > 0 ) {
                    dir = 0;
                    break;
                }
                res.push_back( pts );
                if ( start_new_seq() )
                    break;
                return res;
            }

            const auto &edge = faces[ num_edge ];
            if ( edge.nodes[ 1 - dir ].id == ( dir ? pts.back().id : pts.front().id ) ) {
                if ( edge.nodes[ dir ].id == ( dir ? pts.front().id : pts.back().id ) ) { // in a loop ?
                    if ( start_new_seq() )
                        break;
                    return res;
                }
                if ( dir )
                    pts.push_back( edge.nodes[ 1 ] );
                else
                    pts.push_front( edge.nodes[ 0 ] );
                seen.insert( edge.nodes[ dir ].id );
                break;
            }
        }
    }


    return res;
}

template<class TF,int nvi,int dim,class TI,class NodeData>
RecursivePolytop<TF,nvi,dim,TI,NodeData> RecursivePolytop<TF,nvi,dim,TI,NodeData>::plane_cut( Pt orig, Pt normal, const std::function<Node(const Node &,const Node &,TF,TF)> &nf, std::vector<Face> *new_faces ) const {
    if ( ! nf ) {
        TI mid = max_id();
        std::map<std::pair<TI,TI>,Node> corr;
        return plane_cut( orig, normal, [&]( const Node &a, const Node &b, TF sa, TF sb ) {
            std::pair p{ std::min( a.id, b.id ), std::max( a.id, b.id ) };
            if ( corr.count( p ) )
                return corr[ p ];
            Node &res = corr[ p ];
            res.pos = a.pos + sa / ( sa - sb ) * ( b.pos - a.pos );
            res.id = ++mid;
            return res;
        }, new_faces );
    }

    std::vector<typename Face::Face> np1_faces;
    std::vector<Face> cut_faces;
    for( const Face &face : faces )
        if ( Face cut_face = face.plane_cut( orig, normal, nf, &np1_faces ) )
            cut_faces.push_back( cut_face );

    if ( np1_faces.size() ) {
        // if nvi == 2, np1_faces are actually nodes.
        if constexpr ( nvi == 2 ) {
            for( const DN &dn : non_closed_node_seq( cut_faces ) ) {
                Face nf = Face::make_from( { dn.back(), dn.front() } );
                if ( new_faces )
                    new_faces->push_back( nf );
                cut_faces.push_back( nf );
            }
        } else {
            Face nf = Face::make_from( np1_faces );
            if ( new_faces )
                new_faces->push_back( nf );
            cut_faces.push_back( nf );
        }
    }

    return make_from( cut_faces );
}

template<class TF,int dim,class TI,class NodeData>
RecursivePolytop<TF,1,dim,TI,NodeData> RecursivePolytop<TF,1,dim,TI,NodeData>::plane_cut( Pt orig, Pt normal, const std::function<Node(const Node &,const Node &,TF,TF)> &nf, std::vector<Face> *new_faces ) const {
    TF s0 = dot( nodes[ 0 ].pos - orig, normal );
    TF s1 = dot( nodes[ 1 ].pos - orig, normal );
    if ( s0 <= 0 && s1 <= 0 )
        return { { nodes[ 0 ], nodes[ 1 ] } };
    if ( s0 <= 0 && s1 > 0 ) {
        Node nn = nf( nodes[ 0 ], nodes[ 1 ], s0, s1 );
        if ( new_faces )
            new_faces->push_back( nn );
        return { { nodes[ 0 ], nn } };
    }
    if ( s0 > 0 && s1 <= 0 ) {
        Node nn = nf( nodes[ 0 ], nodes[ 1 ], s0, s1 );
        if ( new_faces )
            new_faces->push_back( nn );
        return { { nn, nodes[ 1 ] } };
    }
    return { {} };
}

template<class TF,int nvi_,int dim,class TI,class NodeData>
TF RecursivePolytop<TF,nvi_,dim,TI,NodeData>::measure( const std::vector<Pt> &prev_dirs, TF div ) const {
    TF res = 0;
    for( const Face &face : faces ) {
        std::vector<Pt> new_dirs = prev_dirs;
        new_dirs.push_back( face.nodes[ 0 ].pos - nodes[ 0 ].pos );
        res += face.measure( new_dirs, div * TF( nvi_ ) );
    }
    return res;
}


template<class TF,int dim,class TI,class NodeData>
TF RecursivePolytop<TF,1,dim,TI,NodeData>::measure( const std::vector<Pt> &prev_dirs, TF div ) const {
    std::vector<Pt> new_dirs = prev_dirs;
    new_dirs.push_back( nodes[ 1 ].pos - nodes[ 0 ].pos );
    return determinant( &new_dirs[ 0 ][ 0 ], N<dim>() ) / div;
}


