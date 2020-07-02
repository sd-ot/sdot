#include "../../support/for_each_comb.h"
#include "../../support/StaticRange.h"
#include "../../support/ASSERT.h"
#include "RecursivePolytopImpl.h"
#include "../../support/P.h"

template<class Rp,int nvi>
RecursivePolytopImpl<Rp,nvi>::RecursivePolytopImpl() : next( nullptr ) {
}

template<class Rp>
RecursivePolytopImpl<Rp,1>::RecursivePolytopImpl() : next( nullptr ) {
}

template<class Rp,int nvi> template<class Fu,int n>
void RecursivePolytopImpl<Rp,nvi>::for_each_item_rec( const Fu &fu, N<n> ) const {
    for( const Face &face : faces )
        face.for_each_item_rec( fu, N<n>() );
}

template<class Rp,int nvi> template<class Fu>
void RecursivePolytopImpl<Rp,nvi>::for_each_item_rec( const Fu &fu, N<nvi> ) const {
    fu( *this );
}

template<class Rp,int nvi> template<class Fu>
void RecursivePolytopImpl<Rp,nvi>::for_each_item_rec( const Fu &fu ) const {
    fu( *this );
    for( const Face &face : faces )
        face.for_each_item_rec( fu );
}

template<class Rp> template<class Fu>
void RecursivePolytopImpl<Rp,1>::for_each_item_rec( const Fu &fu, N<1> ) const {
    fu( *this );
}

template<class Rp> template<class Fu>
void RecursivePolytopImpl<Rp,1>::for_each_item_rec( const Fu &fu ) const {
    fu( *this );
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::add_convex_hull( IntrusiveList<RecursivePolytopImpl> &res, Rp &rp, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &normal, const Pt &center ) {
    // try each possible vertex selection to make new faces
    IntrusiveList<Face> faces;
    for_each_comb<TI>( nvi, nb_indices, indices + nb_indices, [&]( TI *chosen_num_indices ) {
        // normal
        Pt orig = rp.vertices[ indices[ chosen_num_indices[ 0 ] ] ].pos;
        for( TI d = 0; d < nvi - 1; ++d )
            normals[ dim - nvi + d ] = rp.vertices[ indices[ chosen_num_indices[ d + 1 ] ] ].pos - orig;
        Pt face_normal = cross_prod( normals );
        normals[ dim - nvi ] = face_normal;

        // test if we already have this face
        for( const Face &face : faces )
            if ( dot( face.normal, orig - face.first_vertex()->pos ) == 0 && colinear( face.normal, face_normal ) )
                return;

        // test in and out points
        bool has_ins = false;
        bool has_out = false;
        for( TI num_indice = 0; num_indice < nb_indices; ++num_indice ) {
            TF d = dot( rp.vertices[ indices[ num_indice ] ].pos - orig, face_normal );
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
            if ( dot( rp.vertices[ indices[ num_indice ] ].pos - orig, face_normal ) == TF( 0 ) )
                new_indices[ new_nb_indices++ ] = indices[ num_indice ];

        // center
        Pt face_center = TF( 0 );
        for( TI i = 0; i < new_nb_indices; ++i )
            face_center += rp.vertices[ new_indices[ i ] ].pos;
        face_center /= TF( new_nb_indices );

        // update of prev_dirs
        dirs[ dim - nvi ] = face_center - center;

        // construct the new face
        Face::add_convex_hull( faces, rp, new_indices, new_nb_indices, normals, dirs, face_normal, face_center );
    } );

    // register the faces
    if ( ! faces.empty() ) {
        RecursivePolytopImpl *nrp = rp.pool.template create<RecursivePolytopImpl>();
        nrp->faces = std::move( faces );
        nrp->normal = normal;

        res.push_front( nrp );
    }
}

template<class Rp>
void RecursivePolytopImpl<Rp,1>::add_convex_hull( IntrusiveList<RecursivePolytopImpl> &res, Rp &rp, TI *indices, TI nb_indices, Pt */*normals*/, Pt *dirs, const Pt &normal, const Pt &center ) {
    if ( ! nb_indices )
        return;

    // find "left" and "right" points
    dirs[ dim - nvi ] = rp.vertices[ indices[ 0 ] ].pos - center;
    TF s = determinant( dirs->data, N<dim>() ), min_s = s, max_s = s;
    Vertex *v0 = rp.vertices.data() + indices[ 0 ];
    Vertex *v1 = rp.vertices.data() + indices[ 0 ];
    for( TI i = 1; i < nb_indices; ++i ) {
        dirs[ dim - nvi ] = rp.vertices[ indices[ i ] ].pos - center;
        TF s = determinant( dirs->data, N<dim>() );
        if ( min_s > s ) { min_s = s; v0 = rp.vertices.data() + indices[ i ]; }
        if ( max_s < s ) { max_s = s; v1 = rp.vertices.data() + indices[ i ]; }
    }

    if ( v0 != v1 ) {
        RecursivePolytopImpl *nrp = rp.pool.template create<RecursivePolytopImpl>();
        nrp->vertices[ 0 ] = v0;
        nrp->vertices[ 1 ] = v1;
        nrp->normal = normal;

        res.push_front( nrp );
    }
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::write_to_stream( std::ostream &os ) const {
    os << " N: " << normal;
}

template<class Rp>
void RecursivePolytopImpl<Rp,1>::write_to_stream( std::ostream &os ) const {
    os << vertices[ 0 ]->num << " " << vertices[ 1 ]->num;
}

template<class Rp,int nvi>
typename RecursivePolytopImpl<Rp,nvi>::Vertex* RecursivePolytopImpl<Rp,nvi>::first_vertex() const {
    return faces.empty() ? nullptr : faces.front().first_vertex();
}

template<class Rp>
typename RecursivePolytopImpl<Rp,1>::Vertex* RecursivePolytopImpl<Rp,1>::first_vertex() const {
    return vertices[ 0 ];
}

template<class Rp,int nvi> template<class B>
void RecursivePolytopImpl<Rp,nvi>::plane_cut( IntrusiveList<RecursivePolytopImpl> &res, Rp &new_rp, const Rp &old_rp, std::vector<Vertex *> &new_vertices, IntrusiveList<Face> &io_faces, const Vertex *&io_vertex, const Pt &cut_normal, B ) const {
    IntrusiveList<typename Face::Face> new_io_faces;
    IntrusiveList<Face> new_faces;
    for( const Face &face : faces )
        face.plane_cut( new_faces, new_rp, old_rp, new_vertices, new_io_faces, io_vertex, cut_normal, N<nvi-1>() );
    if ( new_faces.empty() )
        return;

    // close the faces
    if ( ! new_io_faces.empty() ) {
        //
        TI nb_graphs = new_rp.get_connected_graphs( new_io_faces );
        for( TI num_graph = 0; num_graph < nb_graphs; ++num_graph ) {
            Face *nf = new_rp.pool.template create<Face>();
            new_faces.push_front( nf );

            nf->normal = cut_normal;
            if ( nb_graphs > 1 )
                new_io_faces.move_to_if( nf->faces, [&]( const typename Face::Face &io_face ) { return new_rp.num_graph( io_face.first_vertex() ) == num_graph; } );
            else
                nf->faces = new_io_faces;

            // to create a new face. New face with a != normal ???
            if ( int( dim ) > int( nvi ) ) {
                Face *of = new_rp.pool.template create<Face>();
                io_faces.push_front( of );
                of->faces = nf->faces;
                of->normal = normal;
            }
        }
    }

    //
    RecursivePolytopImpl *nrp = new_rp.pool.template create<RecursivePolytopImpl>();
    nrp->faces = std::move( new_faces );
    nrp->normal = normal;
    res.push_front( nrp );
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::plane_cut( IntrusiveList<RecursivePolytopImpl> &res, Rp &new_rp, const Rp &old_rp, std::vector<Vertex *> &new_vertices, IntrusiveList<Face> &io_faces, const Vertex *&/*io_vertex*/, const Pt &cut_normal, N<2> ) const {
    using Edge = Face;
    using std::min;
    using std::max;

    IntrusiveList<typename Face::Face> new_io_faces;
    const Vertex *io_vertex = nullptr;
    IntrusiveList<Edge> new_edges;
    for( const Edge &edge : faces )
        edge.plane_cut( new_edges, new_rp, old_rp, new_vertices, new_io_faces, io_vertex, cut_normal, N<1>() );
    if ( new_edges.empty() )
        return;

    // close the face
    if ( const Vertex *b = io_vertex ) {
        // make a linked list
        for( const Edge &edge : faces )
            edge.vertices[ 0 ]->next = edge.vertices[ 1 ];

        //
        for( const Vertex *v = b; ; v = v->next ) {
            if ( io_vertex == nullptr ) {
                // io_vertex == nullptr means that we are inside, and we are looking for a io node
                if ( v->next->outside() )
                    io_vertex = v;
            } else if ( v->next->inside() ) {
                TI b0 = min( io_vertex->num, io_vertex->next->num );
                TI b1 = max( io_vertex->num, io_vertex->next->num );
                TI bn = b1 * ( b1 - 1 ) / 2 + b0;

                TI e0 = min( v->num, v->next->num );
                TI e1 = max( v->num, v->next->num );
                TI en = e1 * ( e1 - 1 ) / 2 + e0;

                // edge to close the current face
                Edge *cl_edge = new_rp.pool.template create<Edge>();
                new_edges.push_front( cl_edge );

                cl_edge->vertices[ 0 ] = new_vertices[ bn ];
                cl_edge->vertices[ 1 ] = new_vertices[ en ];
                cl_edge->normal = cut_normal;

                // edge to create a new face
                if ( int( dim ) > int( nvi ) ) {
                    Edge *nf_edge = new_rp.pool.template create<Edge>();
                    io_faces.push_front( nf_edge );

                    nf_edge->vertices[ 0 ] = new_vertices[ en ];
                    nf_edge->vertices[ 1 ] = new_vertices[ bn ];
                    nf_edge->normal = normal;
                }

                io_vertex = nullptr;
            }

            if ( v->next == b || ! v->next )
                break;
        }

    }

    ///
    RecursivePolytopImpl *nrp = new_rp.pool.template create<RecursivePolytopImpl>();
    nrp->normal = normal;
    nrp->faces = std::move( new_edges );
    res.push_front( nrp );
}

template<class Rp>
void RecursivePolytopImpl<Rp,1>::plane_cut( IntrusiveList<RecursivePolytopImpl> &res, Rp &new_rp, const Rp &old_rp, std::vector<Vertex *> &new_vertices, IntrusiveList<Face> &io_faces, const Vertex *&io_vertex, const Pt &cut_normal, N<1> ) const {
    using std::min;
    using std::max;

    // scalar products
    TF s0 = vertices[ 0 ]->tmp_f;
    TF s1 = vertices[ 1 ]->tmp_f;

    // all inside => nothing to do
    if ( s0 <= 0 && s1 <= 0 ) {
        RecursivePolytopImpl *new_edge = new_rp.pool.template create<RecursivePolytopImpl>();
        res.push_front( new_edge );

        new_edge->vertices[ 0 ] = vertices[ 0 ]->tmp_v;
        new_edge->vertices[ 1 ] = vertices[ 1 ]->tmp_v;
        new_edge->normal = normal;

        return;
    }

    // s1 outside
    if ( s0 <= 0 ) {
        RecursivePolytopImpl *new_edge = new_rp.pool.template create<RecursivePolytopImpl>();
        res.push_front( new_edge );

        TI n0 = min( vertices[ 0 ]->num, vertices[ 1 ]->num );
        TI n1 = max( vertices[ 0 ]->num, vertices[ 1 ]->num );
        TI nn = n1 * ( n1 - 1 ) / 2 + n0;

        new_edge->vertices[ 0 ] = vertices[ 0 ]->tmp_v;
        new_edge->vertices[ 1 ] = new_vertices[ nn ];
        new_edge->normal = normal;

        io_vertex = vertices[ 0 ];

        return;
    }

    // s0 outside
    if ( s1 <= 0 ) {
        RecursivePolytopImpl *new_edge = new_rp.pool.template create<RecursivePolytopImpl>();
        res.push_front( new_edge );

        TI n0 = min( vertices[ 0 ]->num, vertices[ 1 ]->num );
        TI n1 = max( vertices[ 0 ]->num, vertices[ 1 ]->num );
        TI nn = n1 * ( n1 - 1 ) / 2 + n0;

        new_edge->vertices[ 0 ] = new_vertices[ nn ];
        new_edge->vertices[ 1 ] = vertices[ 1 ]->tmp_v;
        new_edge->normal = normal;

        return;
    }
}


//template<class Rp,int nvi>
//void RecursivePolytopImpl<Rp,nvi>::plane_cut( BumpPointerPool &pool, IntrusiveList<Face> &new_faces, std::vector<Node *> &new_vertices, TI &date, N<1> ) {

//    //
//    //    auto set_rp = [&]( Vertex *nv0, Vertex *nv1, TI ind_new ) {
//    //        new_rp.vertices = { pool, 2 };
//    //        new_rp.vertices[ 0 ] = nv0;
//    //        new_rp.vertices[ 1 ] = nv1;
//    //        new_rp.center = TF( 1 ) / 2 * ( new_rp.vertices[ 0 ]->node.pos + new_rp.vertices[ 1 ]->node.pos );
//    //        new_rp.normal = normal;

//    //        for( const Face &face : faces ) {
//    //            Face *new_face = pool.create<Face>();
//    //            new_rp.faces.push_front( new_face );

//    //            new_face->vertices = { pool, 1 };
//    //            new_face->vertices[ 0 ] = new_rp.vertices[ face.vertices[ 0 ] != vertices[ 0 ] ];
//    //            new_face->center = new_face->vertices[ 0 ]->node.pos;
//    //            new_face->normal = face.normal;
//    //        }

//    //        if ( ind_new < 2 ) {
//    //            Face *new_face = pool.create<Face>();
//    //            new_faces.push_front( new_face );

//    //            new_face->vertices = { pool, 1 };
//    //            new_face->vertices[ 0 ] = new_rp.vertices[ ind_new ];
//    //            new_face->center = new_face->vertices[ 0 ]->node.pos;
//    //        }
//    //    };

//    //    // all inside
//    //    if ( s0 <= 0 && s1 <= 0 )
//    //        set_rp( vertices[ 0 ]->tmp_v, vertices[ 1 ]->tmp_v, 2 );

//    //    // only n0 inside
//    //    if ( s0 <= 0 && s1 > 0 ) {
//    //        TI n0 = min( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );
//    //        TI n1 = max( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );
//    //        set_rp( vertices[ 0 ]->tmp_v, new_vertices[ n1 * ( n1 - 1 ) / 2 + n0 ], 1 );
//    //    }

//    //    // only n1 inside
//    //    if ( s0 > 0 && s1 <= 0 ) {
//    //        TI n0 = min( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );
//    //        TI n1 = max( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );
//    //        set_rp( new_vertices[ n1 * ( n1 - 1 ) / 2 + n0 ], vertices[ 1 ]->tmp_v, 0 );
//    //    }
//}

//template<class Rp,int nvi> template<class B>
//void RecursivePolytopImpl<Rp,nvi>::plane_cut( BumpPointerPool &pool, IntrusiveList<Face> &new_faces, std::vector<Node *> &new_vertices, TI &date, B ) {
//    //    IntrusiveList<typename Face::Face> new_new_faces;
//    //    faces.remove_if( [&]( Face &face ) {
//    //        face.plane_cut( pool, new_new_faces, new_vertices, date, N<nvi-1>() );
//    //        return face.faces.empty();
//    //    } );

//    //    if ( ! new_new_faces.empty() ) {
//    //        // new face to close new_rp
//    //        Face *new_face_i = pool.create<Face>();
//    //        new_rp.faces.push_front( new_face_i );

//    //        new_face_i->faces = new_new_faces;

//    //        TI tmp_date = date;
//    //        new_face_i->make_vertices_from_face( pool, tmp_date );
//    //        for( Vertex *v : new_face_i->vertices )
//    //            v->date = date;

//    //        // new face to close new_rp
//    //        if ( int( nvi ) < dim ) {
//    //            Face *new_face_o = pool.create<Face>();
//    //            new_faces.push_front( new_face_o );

//    //            new_face_o->faces = new_new_faces;

//    //            TI tmp_date = date;
//    //            new_face_o->make_vertices_from_face( pool, tmp_date );
//    //            for( Vertex *v : new_face_o->vertices )
//    //                v->date = date;
//    //        }
//    //    }

//    //    // update vertices for new_rp
//    //    TI tmp_date = date;
//    //    new_rp.make_vertices_from_face( pool, tmp_date );
//    //    for( Vertex *v : new_rp.vertices )
//    //        v->date = date;
//}

template<class Rp,int nvi>
typename Rp::TF RecursivePolytopImpl<Rp,nvi>::measure( std::array<Pt,dim> &dirs, const Pt &prev_pt ) const {
    TF res = 0;
    for( const Face &face : faces ) {
        Pt next_pt = face.first_vertex()->pos;
        dirs[ dim - nvi ] = next_pt - prev_pt;
        res += face.measure( dirs, next_pt );
    }
    return res;
}

template<class Rp>
typename Rp::TF RecursivePolytopImpl<Rp,1>::measure( std::array<Pt,dim> &dirs, const Pt &/*prev_pt*/ ) const {
    dirs[ dim - nvi ] = vertices[ 1 ]->pos - vertices[ 0 ]->pos;
    return determinant( dirs[ 0 ].data, N<dim>() );
}

template<class Rp,int nvi>
typename RecursivePolytopImpl<Rp,nvi>::Pt RecursivePolytopImpl<Rp,nvi>::center() const {
    Pt res = TF( 0 );
    TF n = 0;
    for( const Face &face : faces ) {
        res += face.center();
        ++n;
    }
    return n ? res / n : res;
}

template<class Rp>
typename RecursivePolytopImpl<Rp,1>::Pt RecursivePolytopImpl<Rp,1>::center() const {
    return TF( 1 ) / 2 * ( vertices[ 0 ]->pos + vertices[ 1 ]->pos );
}

template<class Rp,int nvi>
bool RecursivePolytopImpl<Rp,nvi>::contains( const Pt &pos ) const {
    while ( true ) {
        // proposition of a direction
        Pt dir;
        for( TI d = 0; d < dim; ++d )
            dir[ d ] = rand() % 10000 - 5000;

        dir = { 1, 0 };
        //
        P( dir );
        for( const Face &face : faces ) {
            face.for_each_intersection( pos, dir, [&]( TF alpha, Pt inter ) {
                P( alpha, inter );
            } );
        }

        break;
    }
    return false;
}

template<class Rp>
bool RecursivePolytopImpl<Rp,1>::contains( const Pt &pos ) const {
    TF s = dot( pos - vertices[ 0 ]->pos, vertices[ 1 ]->pos - vertices[ 0 ]->pos );
    return s >= 0 && s <= norm_2_p2( vertices[ 1 ]->pos - vertices[ 0 ]->pos );
}


template<class Rp>
void RecursivePolytopImpl<Rp,1>::for_each_intersection( const Pt &pos, const Pt &dir, const std::function<void( TF alpha, Pt inter )> &f ) const {
    constexpr int n = nvi + 1;
    std::array<std::array<TF,n>,n> M;
    std::array<TF,n> V;

    Pt tra = vertices[ 0 ]->pos - vertices[ 1 ]->pos;
    M[ 0 ][ 0 ] = dot( dir, dir );
    M[ 0 ][ 1 ] = dot( dir, tra );
    M[ 1 ][ 0 ] = dot( tra, dir );
    M[ 1 ][ 1 ] = dot( tra, tra );
    V[ 0 ] = dot( dir, vertices[ 0 ]->pos - pos );
    V[ 1 ] = dot( tra, vertices[ 0 ]->pos - pos );

    bool ok = true;
    std::array<TF,n> X = solve( M, V, &ok );
    if ( ok )
        f( X[ 0 ], pos + X[ 0 ] * dir );
}


