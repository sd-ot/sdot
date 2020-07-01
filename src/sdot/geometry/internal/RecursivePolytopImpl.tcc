#include "../../support/for_each_comb.h"
#include "../../support/StaticRange.h"
#include "../../support/ASSERT.h"
#include "RecursivePolytopImpl.h"
#include "../../support/P.h"

template<class Rp,int nvi>
RecursivePolytopImpl<Rp,nvi>::RecursivePolytopImpl() : next( nullptr ) {
}

template<class Rp,int nvi> template<class Fu>
void RecursivePolytopImpl<Rp,nvi>::for_each_item_rec( const Fu &fu ) const {
    fu( *this );
    for( const Face &face : faces )
        face.for_each_item_rec( fu );
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::add_convex_hull( BumpPointerPool &pool, const Node *nodes, TI *indices, TI nb_indices, Pt */*prev_normals*/, TI &/*date*/, N<0> ) {
    user_data = nodes[ indices[ 0 ] ].user_data;
    center = nodes[ indices[ 0 ] ].pos;
}

template<class Rp,int nvi> template<class B>
void RecursivePolytopImpl<Rp,nvi>::add_convex_hull( BumpPointerPool &pool, const Node *nodes, TI *indices, TI nb_indices, Pt *prev_normals, TI &date, B ) {
    // try each possible face
    for_each_comb<TI>( nvi, nb_indices, indices + nb_indices, [&]( TI *chosen_num_indices ) {
        // normal in local coordinates
        Pt orig = nodes[ indices[ chosen_num_indices[ 0 ] ] ].pos;
        for( TI d = 0; d < nvi - 1; ++d )
            prev_normals[ dim - nvi + d ] = nodes[ indices[ chosen_num_indices[ d + 1 ] ] ].pos - orig;
        Pt normal = cross_prod( prev_normals );
        prev_normals[ dim - nvi ] = normal;

        // test if we already have this face
        for( const Face &face : faces )
            if ( dot( face.normal, orig - face.center ) == 0 && colinear( face.normal, normal ) )
                return;

        // test in and out points
        bool has_ins = false;
        bool has_out = false;
        for( TI num_indice = 0; num_indice < nb_indices; ++num_indice ) {
            TF d = dot( nodes[ indices[ num_indice ] ].pos - orig, normal );
            has_ins |= d < 0;
            has_out |= d > 0;
        }

        if ( has_ins && has_out )
            return;

        // update normal orientation if necessary
        if ( has_out )
            normal *= TF( -1 );

        // register the new face
        Face *face = pool.create<Face>();
        faces.push_front( face );
        // face->prev_centers;
        face->normal = normal;

        // find all the points that belong to this face
        TI *new_indices = indices + nb_indices + nvi, new_nb_indices = 0;
        for( TI num_indice = 0; num_indice < nb_indices; ++num_indice )
            if ( dot( nodes[ indices[ num_indice ] ].pos - orig, normal ) == TF( 0 ) )
                new_indices[ new_nb_indices++ ] = indices[ num_indice ];

        // construct the new face
        face->add_convex_hull( pool, nodes, new_indices, new_nb_indices, prev_normals, date, N<nvi-1>() );
    } );

    // update center
    TI nb_faces = 0;
    center = TF( 0 );
    for( const Face &face : faces ) {
        center += face.center;
        ++nb_faces;
    }
    if ( nb_faces )
        center /= TF( nb_faces );
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::write_to_stream( std::ostream &os ) const {
    os << "center: "  << center;
    os << " normal: " << normal;
    os << " user_data:" << user_data;
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::sort_vertices( std::array<Pt,dim> &dirs, N<1> ) {
    //    for( Vertex *v : vertices ) {
    //        dirs[ dim - nvi ] = v->node.pos - center;
    //        v->tmp_f = determinant( dirs[ 0 ].data, N<dim>() );
    //    }
    //    std::sort( vertices.begin(), vertices.end(), [&]( Vertex *a, Vertex *b ) {
    //        return a->tmp_f < b->tmp_f;
    //    } );
}


template<class Rp,int nvi> template<class B>
void RecursivePolytopImpl<Rp,nvi>::sort_vertices( std::array<Pt,dim> &dirs, B ) {
    //    for( Face &face : faces ) {
    //        dirs[ dim - nvi ] = face.center - center;
    //        face.sort_vertices( dirs, N<nvi-1>() );
    //    }

    //    if ( nvi == 2 && ! faces.empty() ) {
    //        // make a linked list
    //        for( Face &edge : faces ) {
    //            ASSERT( edge.vertices.size() == 2 );
    //            edge.vertices[ 0 ]->tmp_v = edge.vertices[ 1 ];
    //        }

    //        // make the vertices list with the obtained linked list
    //        TI n = 0;
    //        for( Vertex *e = faces.begin()->vertices[ 0 ], *v = e; ; v = v->tmp_v ) {
    //            vertices[ n++ ] = v;
    //            if ( v->tmp_v == e )
    //                break;
    //        }
    //    }
}

template<class Rp,int nvi>
bool RecursivePolytopImpl<Rp,nvi>::only_outside_points( TI date ) const {
    //    for( Vertex *v : vertices )
    //        if ( v->tmp_f <= 0 && v->date == date )
    //            return false;
    return true;
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::plane_cut( BumpPointerPool &pool, IntrusiveList<Face> &new_faces, std::vector<Node *> &new_vertices, TI &date, N<1> ) {
    //    using std::min;
    //    using std::max;

    //    // scalar products
    //    TF s0 = vertices[ 0 ]->tmp_f;
    //    TF s1 = vertices[ 1 ]->tmp_f;

    //    // all inside => nothing to do
    //    if ( s0 <= 0 && s1 <= 0 )
    //        return;

    //    // all outside
    //    if ( s0 > 0 && s1 > 0 ) {
    //        vertices.clear();
    //        faces.clear();
    //        return;
    //    }

    //
    //    auto set_rp = [&]( Vertex *nv0, Vertex *nv1, TI ind_new ) {
    //        new_rp.vertices = { pool, 2 };
    //        new_rp.vertices[ 0 ] = nv0;
    //        new_rp.vertices[ 1 ] = nv1;
    //        new_rp.center = TF( 1 ) / 2 * ( new_rp.vertices[ 0 ]->node.pos + new_rp.vertices[ 1 ]->node.pos );
    //        new_rp.normal = normal;

    //        for( const Face &face : faces ) {
    //            Face *new_face = pool.create<Face>();
    //            new_rp.faces.push_front( new_face );

    //            new_face->vertices = { pool, 1 };
    //            new_face->vertices[ 0 ] = new_rp.vertices[ face.vertices[ 0 ] != vertices[ 0 ] ];
    //            new_face->center = new_face->vertices[ 0 ]->node.pos;
    //            new_face->normal = face.normal;
    //        }

    //        if ( ind_new < 2 ) {
    //            Face *new_face = pool.create<Face>();
    //            new_faces.push_front( new_face );

    //            new_face->vertices = { pool, 1 };
    //            new_face->vertices[ 0 ] = new_rp.vertices[ ind_new ];
    //            new_face->center = new_face->vertices[ 0 ]->node.pos;
    //        }
    //    };

    //    // all inside
    //    if ( s0 <= 0 && s1 <= 0 )
    //        set_rp( vertices[ 0 ]->tmp_v, vertices[ 1 ]->tmp_v, 2 );

    //    // only n0 inside
    //    if ( s0 <= 0 && s1 > 0 ) {
    //        TI n0 = min( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );
    //        TI n1 = max( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );
    //        set_rp( vertices[ 0 ]->tmp_v, new_vertices[ n1 * ( n1 - 1 ) + n0 ], 1 );
    //    }

    //    // only n1 inside
    //    if ( s0 > 0 && s1 <= 0 ) {
    //        TI n0 = min( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );
    //        TI n1 = max( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );
    //        set_rp( new_vertices[ n1 * ( n1 - 1 ) + n0 ], vertices[ 1 ]->tmp_v, 0 );
    //    }
}

template<class Rp,int nvi> template<class B>
void RecursivePolytopImpl<Rp,nvi>::plane_cut( BumpPointerPool &pool, IntrusiveList<Face> &new_faces, std::vector<Node *> &new_vertices, TI &date, B ) {
    //    IntrusiveList<typename Face::Face> new_new_faces;
    //    faces.remove_if( [&]( Face &face ) {
    //        face.plane_cut( pool, new_new_faces, new_vertices, date, N<nvi-1>() );
    //        return face.faces.empty();
    //    } );

    //    if ( ! new_new_faces.empty() ) {
    //        // new face to close new_rp
    //        Face *new_face_i = pool.create<Face>();
    //        new_rp.faces.push_front( new_face_i );

    //        new_face_i->faces = new_new_faces;

    //        TI tmp_date = date;
    //        new_face_i->make_vertices_from_face( pool, tmp_date );
    //        for( Vertex *v : new_face_i->vertices )
    //            v->date = date;

    //        // new face to close new_rp
    //        if ( int( nvi ) < dim ) {
    //            Face *new_face_o = pool.create<Face>();
    //            new_faces.push_front( new_face_o );

    //            new_face_o->faces = new_new_faces;

    //            TI tmp_date = date;
    //            new_face_o->make_vertices_from_face( pool, tmp_date );
    //            for( Vertex *v : new_face_o->vertices )
    //                v->date = date;
    //        }
    //    }

    //    // update vertices for new_rp
    //    TI tmp_date = date;
    //    new_rp.make_vertices_from_face( pool, tmp_date );
    //    for( Vertex *v : new_rp.vertices )
    //        v->date = date;
}

template<class Rp,int nvi>
typename Rp::TF RecursivePolytopImpl<Rp,nvi>::measure( std::array<Pt,dim> &dirs, N<0> ) const {
    TF d = determinant( dirs[ 0 ].data, N<dim>() );
    return dot( cross_prod( dirs.data() ), normal ) > 0 ? d : - d;
}


template<class Rp,int nvi> template<class B>
typename Rp::TF RecursivePolytopImpl<Rp,nvi>::measure( std::array<Pt,dim> &dirs, B ) const {
    TF res = 0;
    for( const Face &face : faces ) {
        dirs[ dim - nvi ] = face.center - center;
        res += face.measure( dirs, N<nvi-1>() );
    }
    return res;
}

