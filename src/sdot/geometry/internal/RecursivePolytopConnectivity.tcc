#include "../../support/for_each_comb.h"
#include "../../support/ASSERT.h"
#include "../../support/TODO.h"
#include "../../support/P.h"

#include "RecursivePolytopConnectivity.h"

// write_to_stream -----------------------------------------------------------------------
template<class TF,int dim,int nvi,class TI>
void RecursivePolytopConnectivity<TF,dim,nvi,TI>::write_to_stream( std::ostream &os, bool rec ) const {
    if ( rec ) {
        os << "[";
        int cpt = 0;
        for( const Face &face : faces )
            face.write_to_stream( os << ( cpt++ ? "," : "" ), rec );
        os << "]";
    } else
        os << " N: " << normal;
}

template<class TF,int dim,class TI>
void RecursivePolytopConnectivity<TF,dim,1,TI>::write_to_stream( std::ostream &os, bool rec ) const {
    os << "[" << node_numbers[ 0 ] << " " << node_numbers[ 1 ] << "]";
    if ( ! rec )
        os << " N: " << normal;
}

// write_to_stream -----------------------------------------------------------------------
template<class TF,int dim,int nvi,class TI>
TI RecursivePolytopConnectivity<TF,dim,nvi,TI>::first_vertex() const {
    return faces[ 0 ].first_vertex();
}

template<class TF,int dim,class TI>
TI RecursivePolytopConnectivity<TF,dim,1,TI>::first_vertex() const {
    return node_numbers[ 0 ];
}

// for_each_item_rec ---------------------------------------------------------------------
template<class TF,int dim,int nvi,class TI>
 template<class Fu,int n>
void RecursivePolytopConnectivity<TF,dim,nvi,TI>::for_each_item_rec( const Fu &fu, N<n> ) const {
    for( const Face &face : faces )
        face.for_each_item_rec( fu, N<n>() );
}

template<class TF,int dim,int nvi,class TI> template<class Fu>
void RecursivePolytopConnectivity<TF,dim,nvi,TI>::for_each_item_rec( const Fu &fu, N<nvi> ) const {
    fu( *this );
}

template<class TF,int dim,int nvi,class TI> template<class Fu>
void RecursivePolytopConnectivity<TF,dim,nvi,TI>::for_each_item_rec( const Fu &fu ) const {
    fu( *this );
    for( const Face &face : faces )
        face.for_each_item_rec( fu );
}

template<class TF,int dim,class TI> template<class Fu>
void RecursivePolytopConnectivity<TF,dim,1,TI>::for_each_item_rec( const Fu &fu, N<1> ) const {
    fu( *this );
}

template<class TF,int dim,class TI> template<class Fu>
void RecursivePolytopConnectivity<TF,dim,1,TI>::for_each_item_rec( const Fu &fu ) const {
    fu( *this );
}

// set_convex_hull --------------------------------------------------------------------------------------------------
template<class TF,int dim,int nvi,class TI>
void RecursivePolytopConnectivity<TF,dim,nvi,TI>::set_convex_hull( const Pt *points, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &normal, const Pt &center ) {
    //
    this->normal = normal;

    // try each possible vertex selection to make new faces
    for_each_comb<TI>( nvi, nb_indices, indices + nb_indices, [&]( TI *chosen_num_indices ) {
        // normal
        Pt orig = points[ indices[ chosen_num_indices[ 0 ] ] ];
        for( TI d = 0; d < nvi - 1; ++d )
            normals[ dim - nvi + d ] = points[ indices[ chosen_num_indices[ d + 1 ] ] ] - orig;
        Pt face_normal = cross_prod( normals );
        normals[ dim - nvi ] = face_normal;

        // test if we already have this face
        for( const Face &face : faces )
            if ( dot( face.normal, orig - points[ face.first_vertex() ] ) == 0 && colinear( face.normal, face_normal ) )
                return;

        // test in and out points
        bool has_ins = false;
        bool has_out = false;
        for( TI num_indice = 0; num_indice < nb_indices; ++num_indice ) {
            TF d = dot( points[ indices[ num_indice ] ] - orig, face_normal );
            has_ins |= d < 0;
            has_out |= d > 0;
        }

        if ( has_ins && has_out )
            return;

        // update normal orientation if necessary
        if ( has_out )
            face_normal *= TF( -1 );

        // find all the points that belong to this face
        TI *new_indices = indices + nb_indices + nvi, new_nb_indices = 0;
        for( TI num_indice = 0; num_indice < nb_indices; ++num_indice )
            if ( dot( points[ indices[ num_indice ] ] - orig, face_normal ) == TF( 0 ) )
                new_indices[ new_nb_indices++ ] = indices[ num_indice ];

        // center
        Pt face_center = TF( 0 );
        for( TI i = 0; i < new_nb_indices; ++i )
            face_center += points[ new_indices[ i ] ];
        face_center /= TF( new_nb_indices );

        // update of prev_dirs
        dirs[ dim - nvi ] = face_center - center;

        // construct the new face
        Face face;
        face.set_convex_hull( points, new_indices, new_nb_indices, normals, dirs, face_normal, face_center );
        if ( face )
            faces.push_back( std::move( face ) );
    } );
}

template<class TF,int dim,class TI>
void RecursivePolytopConnectivity<TF,dim,1,TI>::set_convex_hull( const Pt *points, TI *indices, TI nb_indices, Pt */*normals*/, Pt *dirs, const Pt &normal, const Pt &center ) {
    if ( ! nb_indices )
        return;

    // find "left" and "right" points
    TI v0 = indices[ 0 ], v1 = indices[ 0 ];
    dirs[ dim - nvi ] = points[ indices[ 0 ] ] - center;
    TF s = determinant( dirs->data, N<dim>() ), min_s = s, max_s = s;
    for( TI i = 1; i < nb_indices; ++i ) {
        dirs[ dim - nvi ] = points[ indices[ i ] ] - center;
        TF s = determinant( dirs->data, N<dim>() );
        if ( min_s > s ) { min_s = s; v0 = indices[ i ]; }
        if ( max_s < s ) { max_s = s; v1 = indices[ i ]; }
    }

    if ( v0 != v1 ) {
        node_numbers = { v0, v1 };
        this->normal = normal;
    }
}

// contains ------------------------------------------------------------------------------------
template<class TF,int dim,int nvi,class TI>
bool RecursivePolytopConnectivity<TF,dim,nvi,TI>::contains( const Pt *points, const Pt &pos ) const {
    for( const Face &face : faces )
        if ( dot( pos - points[ face.first_vertex() ], face.normal ) > 0 )
            return false;
    return true;
}

template<class TF,int dim,class TI>
bool RecursivePolytopConnectivity<TF,dim,1,TI>::contains( const Pt *points, const Pt &pos ) const {
    TF s = dot( pos - points[ node_numbers[ 0 ] ], points[ node_numbers[ 1 ] ] - points[ node_numbers[ 0 ] ] );
    return s >= 0 && s <= norm_2_p2( points[ node_numbers[ 1 ] ] - points[ node_numbers[ 0 ] ] );
}

// center ------------------------------------------------------------------------------------
template<class TF,int dim,int nvi,class TI>
typename RecursivePolytopConnectivity<TF,dim,nvi,TI>::Pt RecursivePolytopConnectivity<TF,dim,nvi,TI>::center( const Pt *points ) const {
    Pt res = TF( 0 );
    TF n = 0;
    for( const Face &face : faces ) {
        res += face.center( points );
        ++n;
    }
    return n ? res / n : res;
}

template<class TF,int dim,class TI>
typename RecursivePolytopConnectivity<TF,dim,1,TI>::Pt RecursivePolytopConnectivity<TF,dim,1,TI>::center( const Pt *points ) const {
    return TF( 1 ) / 2 * ( points[ node_numbers[ 0 ] ] + points[ node_numbers[ 1 ] ] );
}

// measure -----------------------------------------------------------------------------------------------------------------------------
template<class TF,int dim,int nvi,class TI>
TF RecursivePolytopConnectivity<TF,dim,nvi,TI>::measure( const Pt *points, std::array<Pt,dim> &dirs, const Pt &prev_pt ) const {
    TF res = 0;
    for( const Face &face : faces ) {
        Pt next_pt = points[ face.first_vertex() ];
        dirs[ dim - nvi ] = next_pt - prev_pt;
        res += face.measure( points, dirs, next_pt );
    }
    return res;
}

template<class TF,int dim,class TI>
TF RecursivePolytopConnectivity<TF,dim,1,TI>::measure( const Pt *points, std::array<Pt,dim> &dirs, const Pt &/*prev_pt*/ ) const {
    dirs[ dim - nvi ] = points[ node_numbers[ 1 ] ] - points[ node_numbers[ 0 ] ];
    return determinant( dirs[ 0 ].data, N<dim>() );
}

// measure -----------------------------------------------------------------------------------------------------------------------------
template<class TF,int dim,int nvi,class TI> template<int n>
void RecursivePolytopConnectivity<TF,dim,nvi,TI>::conn_cut( std::vector<std::vector<RecursivePolytopConnectivity>> &res, const TI *edge_corr, const TF *sp, N<n> ) const {
    TODO;
}

template<class TF,int dim,int nvi,class TI>
void RecursivePolytopConnectivity<TF,dim,nvi,TI>::conn_cut( std::vector<std::vector<RecursivePolytopConnectivity>> &res, const TI *edge_corr, const TF *sp, N<2> ) const {
    using Edge = Face;
    using std::min;
    using std::max;

    IntrusiveList<typename Face::Face> new_io_faces; // tmp data (not used at all)
    IntrusiveList<Edge> all_the_new_edges; // list that will be used for res
    Vertex *oi_vertices = nullptr; // outside -> inside node list
    for( const Edge &edge : faces )
        edge.plane_cut( all_the_new_edges, new_rp, old_rp, new_vertices, new_io_faces, oi_vertices, cut_normal, N<1>() );
    if ( all_the_new_edges.empty() )
        return;

    // if some edges are cut, we have to close the face(s)
    if ( const Vertex *b = oi_vertices ) {
        // make a linked list of nodes
        for( Edge &edge : all_the_new_edges ) {
            edge.vertices[ 0 ]->next = edge.vertices[ 1 ];
            edge.vertices[ 0 ]->t = &edge;
        }

        // for each output -> input vertex
        for( Vertex *b = oi_vertices; b; b = b->prev_oi ) {
            // find the end of the face
            IntrusiveList<Edge> new_edges;
            Vertex *e = b;
            while ( e->next ) {
                new_edges.push_front( reinterpret_cast<Edge *>( e->t ) );
                e = e->next;
            }

            // edge to close the current face
            Edge *cl_edge = new_rp.pool.template create<Edge>();
            new_edges.push_front( cl_edge );

            cl_edge->vertices[ 0 ] = e;
            cl_edge->vertices[ 1 ] = b;
            cl_edge->normal = cut_normal;

            /// creation of the new face
            RecursiveConvexPolytopImpl *nrp = new_rp.pool.template create<RecursiveConvexPolytopImpl>();
            nrp->faces = std::move( new_edges );
            nrp->normal = normal;
            res.push_front( nrp );

            // edge to create a new face
            if ( int( dim ) > int( nvi ) ) {
                Edge *nf_edge = new_rp.pool.template create<Edge>();
                io_faces.push_front( nf_edge );

                nf_edge->vertices[ 0 ] = b;
                nf_edge->vertices[ 1 ] = e;
                nf_edge->normal = normal;
            }
        }

        return;
    }

    // else, all inside
    RecursiveConvexPolytopImpl *nrp = new_rp.pool.template create<RecursiveConvexPolytopImpl>();
    nrp->faces = std::move( all_the_new_edges );
    nrp->normal = normal;
    res.push_front( nrp );
}

template<class TF,int dim,class TI>
void RecursivePolytopConnectivity<TF,dim,1,TI>::conn_cut( std::vector<std::vector<RecursivePolytopConnectivity>> &res, const TI *edge_corr, const TF *sp, N<1> ) const {
    if ( node_numbers.size() != 2 )
        return;

    bool o0 = sp[ node_numbers[ 0 ] ] > 0;
    bool o1 = sp[ node_numbers[ 1 ] ] > 0;

    if ( o0 && o1 )
        return;

    auto new_point = [&]() {
        TI n0 = std::min( node_numbers[ 0 ], node_numbers[ 1 ] );
        TI n1 = std::max( node_numbers[ 0 ], node_numbers[ 1 ] );
        TI nn = n1 * ( n1 - 1 ) / 2 + n0;
        return edge_corr[ nn ];
    };

    if ( o0 ) {
        RecursivePolytopConnectivity nrp;
        nrp.node_numbers = { new_point(), node_numbers[ 1 ] };
        nrp.normal = TF( 0 );
        res.push_back( { nrp } );
        return;
    }

    if ( o1 ) {
        RecursivePolytopConnectivity nrp;
        nrp.node_numbers = { node_numbers[ 0 ], new_point() };
        nrp.normal = TF( 0 );
        res.push_back( { nrp } );
        return;
    }

    RecursivePolytopConnectivity nrp;
    nrp.node_numbers = { node_numbers[ 0 ], node_numbers[ 1 ] };
    nrp.normal = TF( 0 );
    res.push_back( { nrp } );
}

//template<class TF,int dim,int nvi,class TI>
//void RecursiveConvexPolytopImpl<Rp,nvi>::update_normals( Pt *normals, const Vertex *vertices, TI *indices, const Pt &center ) {
//    // list of unique vertices (in vertices[ . ].tmp_v)
//    TI nb_vertices = make_unique_vertices( vertices );
//    if ( nb_vertices == 0 )
//        return;

//    // find the "best" normal
//    TF best_score = -1;
//    Pt orig = vertices[ 0 ].tmp_v->pos;
//    for_each_comb<TI>( nvi, nb_vertices - 1, indices, [&]( TI *chosen_num_indices ) {
//        for( TI d = 0; d < nvi; ++d )
//            normals[ dim - nvi - 1 + d ] = vertices[ indices[ chosen_num_indices[ d ] ] + 1 ].tmp_v->pos - orig;
//        Pt prop = cross_prod( normals );
//        TF score = norm_2_p2( prop );
//        if ( best_score < score ) {
//            best_score = score;
//            normal = prop;
//        }
//    } );

//    // check orientation (works only for convex polytops)
//    if ( dot( orig - center, normal ) < 0 )
//        normal = - normal;

//    //
//    Pt new_center = TF( 0 );
//    for( TI i = 0; i < nb_vertices; ++i )
//        new_center += vertices[ i ].tmp_v->pos;
//    new_center /= TF( nb_vertices );

//    // next faces
//    normals[ dim - nvi - 1 ] = normal;
//    for( Face &face : faces )
//        face.update_normals( normals, vertices, indices, new_center );
//}

//template<class TF,int dim,class TI>
//void RecursiveConvexPolytopImpl<Rp,1>::update_normals( Pt *normals, const Vertex *, TI *, const Pt &center ) {
//    normals[ dim - 2 ] = vertices[ 1 ]->pos - vertices[ 0 ]->pos;
//    normal = cross_prod( normals );

//    // check orientation (works only for convex polytops)
//    normal *= TF( 1 - 2 * ( dot( vertices[ 0 ]->pos - center, normal ) < 0 ) );
//}


//template<class TF,int dim,int nvi,class TI>
//void RecursiveConvexPolytopImpl<Rp,nvi>::plane_cut( IntrusiveList<RecursiveConvexPolytopImpl> &res, Rp &new_rp, const Rp &old_rp, std::vector<Vertex *> &new_vertices, IntrusiveList<Face> &io_faces, Vertex *&oi_vertices, const Pt &cut_normal, B ) const {
//    IntrusiveList<typename Face::Face> new_io_faces;
//    IntrusiveList<Face> new_faces;
//    for( const Face &face : faces )
//        face.plane_cut( new_faces, new_rp, old_rp, new_vertices, new_io_faces, oi_vertices, cut_normal, N<nvi-1>() );
//    if ( new_faces.empty() )
//        return;

//    // close the faces
//    if ( ! new_io_faces.empty() ) {
//        //
//        TI nb_graphs = new_rp.get_connected_graphs( new_io_faces );
//        for( TI num_graph = 0; num_graph < nb_graphs; ++num_graph ) {
//            Face *nf = new_rp.pool.template create<Face>();
//            new_faces.push_front( nf );

//            nf->normal = cut_normal;
//            if ( nb_graphs > 1 )
//                new_io_faces.move_to_if( nf->faces, [&]( const typename Face::Face &io_face ) { return new_rp.num_graph( io_face.first_vertex() ) == num_graph; } );
//            else
//                nf->faces = new_io_faces;

//            // to create a new face. New face with a != normal ???
//            if ( int( dim ) > int( nvi ) ) {
//                Face *of = new_rp.pool.template create<Face>();
//                io_faces.push_front( of );
//                of->faces = nf->faces;
//                of->normal = normal;
//            }
//        }
//    }

//    //
//    RecursiveConvexPolytopImpl *nrp = new_rp.pool.template create<RecursiveConvexPolytopImpl>();
//    nrp->faces = std::move( new_faces );
//    nrp->normal = normal;
//    res.push_front( nrp );
//}


//bool RecursiveConvexPolytopImpl<Rp,nvi>::valid_vertex_prop( const std::vector<Pt> &pts ) const {
//    for( const Face &face : faces )
//        if ( ! face.valid_vertex_prop( pts ) )
//            return false;

//    //
//    TI rank = 0;
//    bool all_known = true;
//    std::array<Pt,dim> base;
//    const Vertex *f = nullptr;
//    for_each_vertex( [&]( const Vertex *a ) {
//        // we don't have the coordinates for this vertex
//        if ( a->num >= pts.size() ) {
//            all_known = false;
//            return;
//        }

//        // get the first vertex if not already done
//        if ( ! f ) {
//            f = a;
//            return;
//        }

//        // can we add the point to augment the rank ?
//        if ( rank < dim ) {
//            base[ rank ] = pts[ a->num ] - pts[ f->num ];

//            std::array<std::array<TF,dim>,dim> M;
//            for( TI r = 0; r < dim; ++r )
//                for( TI c = 0; c < dim; ++c )
//                    M[ r ][ c ] = TF( r == c );
//            for( TI r = 0; r <= rank; ++r )
//                for( TI c = 0; c <= rank; ++c )
//                    M[ r ][ c ] = dot( base[ r ], base[ c ] );

//            rank += determinant( M ) != 0;
//        }
//    } );

//    if ( rank > nvi )
//        return false;

//    if ( all_known && rank < nvi )
//        return false;

//    // OK
//    return true;
//}

//template<class TF,int dim,class TI>
//bool RecursiveConvexPolytopImpl<Rp,1>::valid_vertex_prop( const std::vector<Pt> &pts ) const {
//    // same point
//    if ( vertices[ 0 ]->num < pts.size() && vertices[ 1 ]->num < pts.size() && pts[ vertices[ 0 ]->num ] == pts[ vertices[ 1 ]->num ] )
//        return false;

//    //
//    return true;
//}


//template<class TF,int dim,class TI>
//void RecursiveConvexPolytopImpl<Rp,1>::for_each_intersection( const Pt &pos, const Pt &dir, const std::function<void( TF alpha, Pt inter )> &f ) const {
//    constexpr int n = nvi + 1;
//    std::array<std::array<TF,n>,n> M;
//    std::array<TF,n> V;

//    Pt tra = vertices[ 0 ]->pos - vertices[ 1 ]->pos;
//    M[ 0 ][ 0 ] = dot( dir, dir );
//    M[ 0 ][ 1 ] = dot( dir, tra );
//    M[ 1 ][ 0 ] = dot( tra, dir );
//    M[ 1 ][ 1 ] = dot( tra, tra );
//    V[ 0 ] = dot( dir, vertices[ 0 ]->pos - pos );
//    V[ 1 ] = dot( tra, vertices[ 0 ]->pos - pos );

//    bool ok = true;
//    std::array<TF,n> X = solve( M, V, &ok );
//    if ( ok )
//        f( X[ 0 ], pos + X[ 0 ] * dir );
//}


