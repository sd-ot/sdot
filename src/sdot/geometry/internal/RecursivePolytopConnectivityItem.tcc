#include "RecursivePolytopConnectivityItemPool.h"
#include "../../support/for_each_comb.h"
#include "../../support/P.h"

// write_to_stream -----------------------------------------------------------------------
template<class TF,class TI,int nvi>
void RecursivePolytopConnectivityItem<TF,TI,nvi>::write_to_stream( std::ostream &os ) const {
    os << "[";
    for( TI i = 0; i < faces.size(); ++i )
        faces[ i ]->write_to_stream( os << ( i++ ? "," : "" ) );
    os << "]";
}

template<class TF,class TI>
void RecursivePolytopConnectivityItem<TF,TI,0>::write_to_stream( std::ostream &os ) const {
    os << node_number;
}

// New ------------------------------------------------------------------------------------
template<class TF,class TI,int nvi> template<class Pt>
void RecursivePolytopConnectivityItem<TF,TI,nvi>::add_convex_hull( std::vector<Item *> &res, ItemPool &item_pool, BumpPointerPool &mem_pool, const Pt *positions, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &center ) {
    static constexpr int dim = Pt::dim;
    std::vector<Pt> face_normals;
    std::vector<Face *> faces;

    // try each possible vertex selection to make new faces
    for_each_comb<TI>( nvi, nb_indices, indices + nb_indices, [&]( TI *chosen_num_indices ) {
        // normal
        Pt orig = positions[ indices[ chosen_num_indices[ 0 ] ] ];
        for( TI d = 0; d < nvi - 1; ++d )
            normals[ dim - nvi + d ] = positions[ indices[ chosen_num_indices[ d + 1 ] ] ] - orig;
        Pt face_normal = cross_prod( normals );
        normals[ dim - nvi ] = face_normal;

        // test if we already have this face
        for( TI i = 0; i < faces.size(); ++i )
            if ( dot( face_normals[ i ], orig - positions[ faces[ i ]->first_vertex()->node_number ] ) == 0 && colinear( face_normals[ i ], face_normal ) )
                return;

        // test in and out points
        bool has_ins = false;
        bool has_out = false;
        for( TI num_indice = 0; num_indice < nb_indices; ++num_indice ) {
            TF d = dot( positions[ indices[ num_indice ] ] - orig, face_normal );
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
            if ( dot( positions[ indices[ num_indice ] ] - orig, face_normal ) == TF( 0 ) )
                new_indices[ new_nb_indices++ ] = indices[ num_indice ];

        // center
        Pt face_center = TF( 0 );
        for( TI i = 0; i < new_nb_indices; ++i )
            face_center += positions[ new_indices[ i ] ];
        face_center /= TF( new_nb_indices );

        // update of prev_dirs
        dirs[ dim - nvi ] = face_center - center;

        // add the new face
        Face::add_convex_hull( faces, item_pool.next, mem_pool, positions, new_indices, new_nb_indices, normals, dirs, face_center );
        face_normals.push_back( face_normal );
    } );

    // create a new item
    if ( faces.size() ) {
        std::sort( faces.begin(), faces.end() );
        res.push_back( item_pool.find_or_create( mem_pool, std::move( faces ) ) );
    }
}

template<class TF,class TI> template<class Pt>
void RecursivePolytopConnectivityItem<TF,TI,0>::add_convex_hull( std::vector<Item *> &res, ItemPool &item_pool, BumpPointerPool &mem_pool, const Pt */*positions*/, TI *indices, TI /*nb_indices*/, Pt */*normals*/, Pt */*dirs*/, const Pt &/*center*/ ) {
    res.push_back( item_pool.find_or_create( mem_pool, *indices ) );
}
