#include "../../support/for_each_comb.h"
#include "../../support/StaticRange.h"
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
void RecursivePolytopImpl<Rp,nvi>::add_convex_hull( BumpPointerPool &pool, FsVec<Vertex> vertices, TI *indices, TI nb_indices, Pt *prev_normals, N<0> ) {
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
void RecursivePolytopImpl<Rp,nvi>::add_convex_hull( BumpPointerPool &pool, FsVec<Vertex> vertices, TI *indices, TI nb_indices, Pt *prev_normals, B ) {
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
        face->add_convex_hull( pool, vertices, new_indices, new_nb_indices, prev_normals, N<nvi-1>() );
    } );

    // make the vertex list


    // update center
    center = TF( 0 );
    for( TI num_indice = 0; num_indice < nb_indices; ++num_indice )
        center += vertices[ indices[ num_indice ] ].node.pos;
    center /= TF( nb_indices );
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::write_to_stream( std::ostream &os ) const {
    os << "center: "  << center;
    os << " normal: " << normal;
}


