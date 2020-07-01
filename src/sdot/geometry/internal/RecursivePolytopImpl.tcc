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
void RecursivePolytopImpl<Rp,nvi>::add_convex_hull( BumpPointerPool &pool, FsVec<Vertex> vertices, TI *indices, TI nb_indices, N<0> ) {
    P( FsVec<TI>{ indices, nb_indices } );

    this->vertices = { pool, nb_indices };
    for( TI i = 0; i < nb_indices; ++i )
        this->vertices[ i ] = &vertices[ indices[ i ] ];
}

template<class Rp,int nvi> template<class B>
void RecursivePolytopImpl<Rp,nvi>::add_convex_hull( BumpPointerPool &pool, FsVec<Vertex> vertices, TI *indices, TI nb_indices, B ) {
    make_base_dirs( base_dirs, vertices, indices, nb_indices );

    P( FsVec<TI>{ indices, nb_indices }, base_dirs );

    // try each possible face
    for_each_comb<TI>( nvi, nb_indices, indices + nb_indices, [&]( TI *chosen_num_indices ) {
        // normal in local coordinates
        Pt orig = vertices[ indices[ chosen_num_indices[ 0 ] ] ].node.pos;
        std::array<Pn,nvi-1> loc_dirs;
        for( TI d = 0; d < nvi - 1; ++d )
            loc_dirs[ d ] = proj( vertices[ indices[ chosen_num_indices[ d + 1 ] ] ].node.pos - orig );
        Pn loc_normal = cross_prod( loc_dirs.data() );

        // normal in global coordinates
        Pt normal = TF( 0 );
        for( TI n = 0; n < nvi; ++n )
            normal += loc_normal[ n ] * base_dirs[ n ];
        P( normal );

        // test if we already have this face
        for( const Face &face : faces )
            if ( dot( face.normal, orig - face.orig ) == 0 && colinear( face.normal, normal ) )
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
        // face->prev_centers;
        face->normal = normal;
        face->orig = orig;
        faces.push_front( face );

        // find all the points that belong to this face
        TI *new_indices = indices + nb_indices + nvi, new_nb_indices = 0;
        for( TI num_indice = 0; num_indice < nb_indices; ++num_indice )
            if ( dot( vertices[ indices[ num_indice ] ].node.pos - orig, normal ) == TF( 0 ) )
                new_indices[ new_nb_indices++ ] = indices[ num_indice ];

        // construct the new face
        face->add_convex_hull( pool, vertices, new_indices, new_nb_indices, N<nvi-1>() );
    } );

    // make the vertex list
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::write_to_stream( std::ostream &os ) const {
    os << "base_dirs: " << base_dirs;
    os << " normal: "   << normal;
    os << " orig: "     << orig;
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::make_base_dirs( std::array<Pt,nvi> &base_dirs, FsVec<Vertex> vertices, const TI *indices, TI nb_indices ) {
    StaticRange<nvi>::for_each( [&]( auto d ) {
        using std::abs;

        TF best_determinant = 0;
        base_dirs[ d.value ] = TF( 0 );
        for( TI num_index = 1; num_index < nb_indices; ++num_index ) {
            Pt dir = vertices[ indices[ num_index ] ].node.pos - vertices[ indices[ 0 ] ].node.pos;

            std::array<std::array<TF,d.value+1>,d.value+1> M;
            for( std::size_t r = 1; r < d.value; ++r )
                for( std::size_t c = 0; c < r; ++c )
                    M[ c ][ r ] = M[ r ][ c ] = dot( base_dirs[ r ], base_dirs[ c ] );
            for( std::size_t r = 0; r < d.value; ++r )
                M[ r ][ r ] = dot( base_dirs[ r ], base_dirs[ r ] );
            for( std::size_t r = 0; r < d.value; ++r )
                M[ d.value ][ r ] = M[ r ][ d.value ] = dot( base_dirs[ r ], dir );
            M[ d.value ][ d.value ] = dot( dir, dir );

            TF ad = abs( determinant( M ) );
            if ( best_determinant < ad ) {
                base_dirs[ d.value ] = dir;
                best_determinant = ad;
            }
        }
    } );
}

template<class Rp,int nvi>
typename RecursivePolytopImpl<Rp,nvi>::Pn RecursivePolytopImpl<Rp,nvi>::proj( const Pt &pt ) const {
    std::array<std::array<TF,nvi>,nvi> M;
    std::array<TF,nvi> V;
    for( TI r = 0; r < nvi; ++r )
        for( TI c = 0; c < nvi; ++c )
            M[ r ][ c ] = dot( base_dirs[ r ], base_dirs[ c ] );
    for( TI r = 0; r < nvi; ++r )
        V[ r ] = dot( base_dirs[ r ], pt );

    std::array<TF,nvi> X = solve( M, V );
    return X.data();
}


