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
    std::vector<FaceItem *> face_items;
    std::vector<Pt> face_normals;

    // try each possible vertex selection to make new faces
    for_each_comb<TI>( nvi, nb_indices, indices + nb_indices, [&]( TI *chosen_num_indices ) {
        // normal
        Pt orig = positions[ indices[ chosen_num_indices[ 0 ] ] ];
        for( TI d = 0; d < nvi - 1; ++d )
            normals[ dim - nvi + d ] = positions[ indices[ chosen_num_indices[ d + 1 ] ] ] - orig;
        Pt face_normal = cross_prod( normals );
        normals[ dim - nvi ] = face_normal;

        // test if we already have this face
        for( TI i = 0; i < face_items.size(); ++i )
            if ( dot( face_normals[ i ], orig - positions[ face_items[ i ]->first_vertex()->node_number ] ) == 0 && colinear( face_normals[ i ], face_normal ) )
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
        FaceItem::add_convex_hull( face_items, item_pool.next, mem_pool, positions, new_indices, new_nb_indices, normals, dirs, face_center );
        face_normals.push_back( face_normal );
    } );

    // create a new item
    if ( face_items.size() ) {
        std::vector<Face> new_faces( face_items.size() );
        std::sort( face_items.begin(), face_items.end(), []( Face *a, Face *b ) { return *a < *b; } );
        res.push_back( item_pool.find_or_create( mem_pool, std::move( face_items ) ) );
    }
}

template<class TF,class TI> template<class Pt>
void RecursivePolytopConnectivityItem<TF,TI,0>::add_convex_hull( std::vector<Item *> &res, ItemPool &item_pool, BumpPointerPool &mem_pool, const Pt */*positions*/, TI *indices, TI /*nb_indices*/, Pt */*normals*/, Pt *dirs, const Pt &/*center*/ ) {
    bool is_start = determinant( dirs->data, N<Pt::dim>() ) > 0;
    res.push_back( item_pool.find_or_create( mem_pool, *indices, is_start ) );
}

// copy -----------------------------------------------------------------------
template<class TF,class TI,int nvi> template<class Pt>
RecursivePolytopConnectivityItem<TF,TI,nvi> *RecursivePolytopConnectivityItem<TF,TI,nvi>::copy_rec( std::vector<Pt> &new_positions, ItemPool &new_item_pool, BumpPointerPool &new_mem_pool, const std::vector<Pt> &old_positions ) const {
    if ( ! new_item ) {
        std::vector<Face *> new_faces( faces.size() );
        for( TI i = 0; i < faces.size(); ++i )
            new_faces[ i ] = faces[ i ]->copy_rec( new_positions, new_item_pool.next, new_mem_pool, old_positions );
        std::sort( new_faces.begin(), new_faces.end(), []( Face *a, Face *b ) { return *a < *b; } );
        new_item = new_item_pool.create( new_mem_pool, std::move( new_faces ) );
    }

    return new_item;
}

template<class TF,class TI> template<class Pt>
RecursivePolytopConnectivityItem<TF,TI,0> *RecursivePolytopConnectivityItem<TF,TI,0>::copy_rec( std::vector<Pt> &new_positions, ItemPool &new_item_pool, BumpPointerPool &new_mem_pool, const std::vector<Pt> &old_positions ) const {
    if ( ! new_item ) {
        new_item = new_item_pool.create( new_mem_pool, new_positions.size() );
        new_positions.push_back( old_positions[ node_number ] );
    }

    return new_item;
}
