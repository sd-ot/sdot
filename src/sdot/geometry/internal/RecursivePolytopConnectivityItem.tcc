#include "RecursivePolytopConnectivityItemPool.h"
#include "../../support/for_each_comb.h"

// write_to_stream -----------------------------------------------------------------------
template<class TF,class TI,int nvi>
void RecursivePolytopConnectivityItem<TF,TI,nvi>::write_to_stream( std::ostream &os ) const {
    os << "[";
    for( TI i = 0; i < sorted_faces.size(); ++i )
        sorted_faces[ i ]->write_to_stream( os << ( i++ ? "," : "" ) );
    os << "]";
}

template<class TF,class TI>
void RecursivePolytopConnectivityItem<TF,TI,0>::write_to_stream( std::ostream &os ) const {
    os << node_number;
}

// New ------------------------------------------------------------------------------------
template<class TF,class TI,int nvi> template<class Pt>
void RecursivePolytopConnectivityItem<TF,TI,nvi>::add_convex_hull( ItemPool &item_pool, BumpPointerPool &mem_pool, std::vector<Item *> &res, const Pt *positions, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &normal, const Pt &center ) {
    TODO;
    //    // try each possible vertex selection to make new faces
    //    for_each_comb<TI>( nvi, nb_indices, indices + nb_indices, [&]( TI *chosen_num_indices ) {
    //        // normal
    //        Pt orig = points[ indices[ chosen_num_indices[ 0 ] ] ];
    //        for( TI d = 0; d < nvi - 1; ++d )
    //            normals[ dim - nvi + d ] = points[ indices[ chosen_num_indices[ d + 1 ] ] ] - orig;
    //        Pt face_normal = cross_prod( normals );
    //        normals[ dim - nvi ] = face_normal;

    //        // test if we already have this face
    //        for( const Face &face : faces )
    //            if ( dot( face.normal, orig - points[ face.first_vertex() ] ) == 0 && colinear( face.normal, face_normal ) )
    //                return;

    //        // test in and out points
    //        bool has_ins = false;
    //        bool has_out = false;
    //        for( TI num_indice = 0; num_indice < nb_indices; ++num_indice ) {
    //            TF d = dot( points[ indices[ num_indice ] ] - orig, face_normal );
    //            has_ins |= d < 0;
    //            has_out |= d > 0;
    //        }

    //        if ( has_ins && has_out )
    //            return;

    //        // update normal orientation if necessary
    //        if ( has_out )
    //            face_normal *= TF( -1 );

    //        // find all the points that belong to this face
    //        TI *new_indices = indices + nb_indices + nvi, new_nb_indices = 0;
    //        for( TI num_indice = 0; num_indice < nb_indices; ++num_indice )
    //            if ( dot( points[ indices[ num_indice ] ] - orig, face_normal ) == TF( 0 ) )
    //                new_indices[ new_nb_indices++ ] = indices[ num_indice ];

    //        // center
    //        Pt face_center = TF( 0 );
    //        for( TI i = 0; i < new_nb_indices; ++i )
    //            face_center += points[ new_indices[ i ] ];
    //        face_center /= TF( new_nb_indices );

    //        // update of prev_dirs
    //        dirs[ dim - nvi ] = face_center - center;

    //        // construct the new face
    //        Face face;
    //        face.set_convex_hull( points, new_indices, new_nb_indices, normals, dirs, face_normal, face_center );
    //        if ( face )
    //            faces.push_back( std::move( face ) );
    //    } );
}

