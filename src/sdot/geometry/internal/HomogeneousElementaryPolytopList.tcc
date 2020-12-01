#include "HomogeneousElementaryPolytopList.h"

template<class AF,class AI>
HomogeneousElementaryPolytopList<AF,AI>::HomogeneousElementaryPolytopList( AF &allocator_TF, AI &allocator_TI, TI nb_nodes, TI nb_faces, TI dim, TI rese_items ) {
    positions.resize( allocator_TF, nb_nodes, dim, rese_items );
    face_ids.resize( allocator_TI, nb_faces, rese_items );
    ids.resize( allocator_TI, rese_items );
}

template<class AF,class AI>
void HomogeneousElementaryPolytopList<AF,AI>::write_to_stream( std::ostream &os, const AF &af, const AI &ai, const std::string &sp ) const {
    for( TI num_item = 0; num_item < size(); ++num_item ) {
        os << sp;
        for( TI num_node = 0; num_node < nb_nodes(); ++num_node ) {
            if ( num_node )
                os << ", ";
            for( TI d = 0; d < dim(); ++d ) {
                if ( d )
                    os << " ";
                os << positions.at( af, num_node, d, num_item );
            }
        }
        os << ";";
        for( TI num_face = 0; num_face < nb_faces(); ++num_face )
            os << " " << face_ids.at( ai, num_face, num_item );
        os << "; " << ids.at( ai, num_item );
    }
}

template<class AF,class AI>
void HomogeneousElementaryPolytopList<AF,AI>::resize( AF &allocator_TF, AI &allocator_TI, TI new_size ) {
    positions.resize( allocator_TF, nb_nodes(), dim(), new_size );
    face_ids.resize( allocator_TI, nb_faces(), new_size );
    ids.resize( allocator_TI, new_size );
}
