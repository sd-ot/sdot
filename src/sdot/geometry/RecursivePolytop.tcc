#include "../support/for_each_comb.h"
#include "../support/ASSERT.h"
#include "RecursivePolytop.h"
#include <algorithm>

template<class TF,int nvi,int dim,class TI,class NodeData>
RecursivePolytop<TF,nvi,dim,TI,NodeData> RecursivePolytop<TF,nvi,dim,TI,NodeData>::convex_hull( const std::vector<Node> &nodes, const std::vector<Pt> &prev_centers ) {
    struct DFace { std::array<Pt,dim-1> dirs; Pt normal, orig; };
    std::vector<DFace> dfaces;

    // new center
    Pt center = TF( 0 );
    for( const Node &node : nodes )
        center += node.pos;
    center /= TF( nodes.size() );

    // new center list
    std::vector<Pt> centers = prev_centers;
    centers.push_back( center );

    // try each way to make a face
    RecursivePolytop res;
    for_each_comb<TI>( nodes.size(), dim, [&]( std::vector<TI> num_nodes ) {
        // make a face trial (to store it if it's new and relevant)
        DFace dface;
        dface.orig = nodes[ num_nodes[ 0 ] ].pos;
        for( TI d = 0; d < dim - 1; ++d )
            dface.dirs[ d ] = nodes[ num_nodes[ d + 1 ] ].pos - dface.orig;

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
                TF d = dot( nodes[ op ].pos - dface.orig, dface.normal );
                has_ins |= d < 0;
                has_out |= d > 0;
            }
        }
        if ( has_ins && has_out )
            return;

        // update normal orientation if necessary
        if ( has_out )
            dface.normal *= TF( -1 );

        // find all the points that belong to this face
        std::vector<Node> new_nodes;
        for( TI op = 0; op < nodes.size(); ++op )
            if ( dot( nodes[ op ].pos - dface.orig, dface.normal ) == TF( 0 ) )
                new_nodes.push_back( nodes[ op ] );

        // register nodes
        for( const Node &node : new_nodes )
            if ( std::find( res.nodes.begin(), res.nodes.end(), node ) == res.nodes.end() )
                res.nodes.push_back( node );

        // register face
        res.faces.push_back( Face::convex_hull( new_nodes, centers ) );
    } );

    // sort all the faces
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
    os << sp << "+";
    for( const Face &face : faces )
        face.write_to_stream( os, sp + "  " );
}

template<class TF,int dim,class TI,class NodeData>
void RecursivePolytop<TF,1,dim,TI,NodeData>::write_to_stream( std::ostream &os, std::string sp ) const {
    os << sp;
    for( TI i = 0; i < nodes.size(); ++i )
        os << ( i ? "; " : "" ) << nodes[ i ].pos << " " << nodes[ i ].data;
}

template<class TF,int nvi,int dim,class TI,class NodeData>
bool RecursivePolytop<TF,nvi,dim,TI,NodeData>::operator<( const RecursivePolytop &that ) const {
    return faces < that.faces;
}

template<class TF,int dim,class TI,class NodeData>
bool RecursivePolytop<TF,1,dim,TI,NodeData>::operator<( const RecursivePolytop &that ) const {
    return nodes < that.nodes;
}
