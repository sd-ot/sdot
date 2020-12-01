#include "HomogeneousElementaryPolytopList.h"

template<class TF,class TI>
HomogeneousElementaryPolytopList<TF,TI>::HomogeneousElementaryPolytopList( TI nb_nodes, TI nb_faces, TI dim, TI rese_items ) {
    positions.resize( { nb_nodes, dim, rese_items } );
    face_ids.resize( { nb_faces, rese_items } );
    ids.resize( { rese_items } );
}

template<class TF, class TI>
void HomogeneousElementaryPolytopList<TF,TI>::write_to_stream( std::ostream &os, const std::string &sp ) const {
    for( TI num_item = 0; num_item < size(); ++num_item ) {
        os << sp;
        for( TI num_node = 0; num_node < nb_nodes(); ++num_node ) {
            if ( num_node )
                os << ", ";
            for( TI d = 0; d < dim(); ++d ) {
                if ( d )
                    os << " ";
                os << positions( num_node, d, num_item );
            }
        }
        os << ";";
        for( TI num_face = 0; num_face < nb_faces(); ++num_face )
            os << " " << face_ids( num_face, num_item );
        os << "; " << ids( num_item );
    }
}

template<class TF, class TI>
void HomogeneousElementaryPolytopList<TF,TI>::resize( TI new_size ) {
    positions.resize( { nb_nodes(), dim(), new_size } );
    face_ids.resize( { nb_faces(), new_size } );
    ids.resize( { new_size } );
}
