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
void RecursivePolytopImpl<Rp,nvi>::add_convex_hull( BumpPointerPool &pool, FsVec<Vertex> vertices, TI *indices, TI nb_indices, Pt */*prev_normals*/, TI &/*date*/, N<0> ) {
    this->vertices = { pool, nb_indices };
    for( TI i = 0; i < nb_indices; ++i )
        this->vertices[ i ] = &vertices[ indices[ i ] ];

    // update center
    center = TF( 0 );
    for( TI num_indice = 0; num_indice < nb_indices; ++num_indice )
        center += vertices[ indices[ num_indice ] ].node.pos;
    center /= TF( nb_indices );
}

template<class Rp,int nvi> template<class B>
void RecursivePolytopImpl<Rp,nvi>::add_convex_hull( BumpPointerPool &pool, FsVec<Vertex> vertices, TI *indices, TI nb_indices, Pt *prev_normals, TI &date, B ) {
    // try each possible face
    for_each_comb<TI>( nvi, nb_indices, indices + nb_indices, [&]( TI *chosen_num_indices ) {
        // normal in local coordinates
        Pt orig = vertices[ indices[ chosen_num_indices[ 0 ] ] ].node.pos;
        for( TI d = 0; d < nvi - 1; ++d )
            prev_normals[ dim - nvi + d ] = vertices[ indices[ chosen_num_indices[ d + 1 ] ] ].node.pos - orig;
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
            TF d = dot( vertices[ indices[ num_indice ] ].node.pos - orig, normal );
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
            if ( dot( vertices[ indices[ num_indice ] ].node.pos - orig, normal ) == TF( 0 ) )
                new_indices[ new_nb_indices++ ] = indices[ num_indice ];

        // construct the new face
        face->add_convex_hull( pool, vertices, new_indices, new_nb_indices, prev_normals, date, N<nvi-1>() );
    } );

    // make the vertex list
    make_vertices_from_face( pool, date );

    // update center
    center = TF( 0 );
    for( Vertex *v : this->vertices )
        center += v->node.pos;
    center /= TF( nb_indices );
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::make_vertices_from_face( BumpPointerPool &pool, TI &date ) {
    // set size
    ++date;
    TI nb_vertices = 0;
    for( const Face &face : faces ) {
        for( Vertex *v : face.vertices ) {
            if ( v->date < date ) {
                v->date = date;
                ++nb_vertices;
            }
        }
    }

    vertices = { pool, nb_vertices };

    // set values
    ++date;
    nb_vertices = 0;
    for( const Face &face : faces ) {
        for( Vertex *v : face.vertices ) {
            if ( v->date < date ) {
                v->date = date;
                vertices[ nb_vertices++ ] = v;
            }
        }
    }
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::write_to_stream( std::ostream &os ) const {
    os << "center: "  << center;
    os << " normal: " << normal;
    os << " vertices:";
    for( Vertex *v : vertices )
        os << " [" << v->node.pos << "]";
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::sort_vertices( std::array<Pt,dim> &dirs, N<1> ) {
    for( Vertex *v : vertices ) {
        dirs[ dim - nvi ] = v->node.pos - center;
        v->tmp_f = determinant( dirs[ 0 ].data, N<dim>() );
    }
    std::sort( vertices.begin(), vertices.end(), [&]( Vertex *a, Vertex *b ) {
        return a->tmp_f < b->tmp_f;
    } );
}


template<class Rp,int nvi> template<class B>
void RecursivePolytopImpl<Rp,nvi>::sort_vertices( std::array<Pt,dim> &dirs, B ) {
    for( Face &face : faces ) {
        dirs[ dim - nvi ] = face.center - center;
        face.sort_vertices( dirs, N<nvi-1>() );
    }

    if ( nvi == 2 && ! faces.empty() ) {
        // make a linked list
        for( Face &edge : faces ) {
            ASSERT( edge.vertices.size() == 2 );
            edge.vertices[ 0 ]->tmp_v = edge.vertices[ 1 ];
        }

        // make the vertices list with the obtained linked list
        TI n = 0;
        for( Vertex *e = faces.begin()->vertices[ 0 ], *v = e; ; v = v->tmp_v ) {
            vertices[ n++ ] = v;
            if ( v->tmp_v == e )
                break;
        }
    }
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::plane_cut( BumpPointerPool &pool, IntrusiveList<RecursivePolytopImpl> &new_rps, std::vector<Vertex *> &new_vertices, TI date, N<1> ) const {
    using std::min;
    using std::max;

    TF s0 = vertices[ 0 ]->tmp_f;
    TF s1 = vertices[ 1 ]->tmp_f;

    // all inside
    if ( s0 <= 0 && s1 <= 0 ) {
        RecursivePolytopImpl *new_rp = pool.create<RecursivePolytopImpl>();
        new_rp->vertices = { pool, 2 };
        new_rp->vertices[ 0 ] = vertices[ 0 ]->tmp_v;
        new_rp->vertices[ 1 ] = vertices[ 1 ]->tmp_v;
        new_rp->center = center;
        new_rp->normal = normal;

        new_rps.push_front( new_rp );
    }

    // only n0 inside
    if ( s0 <= 0 && s1 > 0 ) {
        TI n0 = min( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );
        TI n1 = max( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );

        RecursivePolytopImpl *new_rp = pool.create<RecursivePolytopImpl>();
        new_rp->vertices = { pool, 2 };
        new_rp->vertices[ 0 ] = vertices[ 0 ]->tmp_v;
        new_rp->vertices[ 1 ] = new_vertices[ n1 * ( n1 - 1 ) + n0 ];
        new_rp->center = TF( 1 ) / 2 * ( new_rp->vertices[ 0 ]->node.pos + new_rp->vertices[ 1 ]->node.pos );
        new_rp->normal = normal;

        new_rps.push_front( new_rp );
    }

    // only n1 inside
    if ( s0 > 0 && s1 <= 0 ) {
        TI n0 = min( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );
        TI n1 = max( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );

        RecursivePolytopImpl *new_rp = pool.create<RecursivePolytopImpl>();
        new_rp->vertices = { pool, 2 };
        new_rp->vertices[ 0 ] = new_vertices[ n1 * ( n1 - 1 ) + n0 ];
        new_rp->vertices[ 1 ] = vertices[ 1 ]->tmp_v;
        new_rp->center = TF( 1 ) / 2 * ( new_rp->vertices[ 0 ]->node.pos + new_rp->vertices[ 1 ]->node.pos );
        new_rp->normal = normal;

        new_rps.push_front( new_rp );
    }
}

template<class Rp,int nvi> template<class B>
void RecursivePolytopImpl<Rp,nvi>::plane_cut( BumpPointerPool &pool, IntrusiveList<RecursivePolytopImpl> &new_rps, std::vector<Vertex *> &new_vertices, TI date, B ) const {
    for( const Face &face : faces ) {
        IntrusiveList<Face> new_faces;
        face.plane_cut( pool, new_faces, new_vertices, date, N<nvi-1>() );

        if ( ! new_faces.empty() ) {
            RecursivePolytopImpl *new_rp = pool.create<RecursivePolytopImpl>();
            new_rp->faces = new_faces;
            // new_rp->make_vertices_from_face() -> Pb: ça modifie la date
            // new_rp->vertices = { pool, 2 };
            //            new_rp->vertices[ 0 ] = vertices[ 0 ]->tmp_v;
            //            new_rp->vertices[ 1 ] = vertices[ 1 ]->tmp_v;
            //            new_rp->center = center;
            //            new_rp->normal = normal;

            new_rps.push_front( new_rp );
        }
    }
}

template<class Rp,int nvi>
typename Rp::TF RecursivePolytopImpl<Rp,nvi>::measure( std::array<Pt,dim> &dirs, N<1> ) const {
    if ( vertices.size() < 2 )
        return 0;
    dirs[ dim - nvi ] = vertices[ vertices.size() - 1 ]->node.pos - vertices[ 0 ]->node.pos;
    return determinant( dirs[ 0 ].data, N<dim>() );
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

