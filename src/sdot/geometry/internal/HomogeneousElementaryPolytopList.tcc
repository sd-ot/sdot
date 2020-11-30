#include "HomogeneousElementaryPolytopList.h"

template<class TF,class TI>
void HomogeneousElementaryPolytopList<TF,TI>::write_to_stream( std::ostream &os, const std::string &sp ) const {
    TI nb_nodes = positions.shape()[ 1 ];
    TI nb_faces = face_ids.shape()[ 1 ];
    TI dim = positions.shape()[ 1 ];

    for( TI num_item = 0; num_item < size(); ++num_item ) {
        os << sp;
        for( TI num_node = 0; num_node < nb_nodes; ++num_node ) {
            if ( num_node )
                os << ", ";
            for( TI d = 0; d < dim; ++d ) {
                if ( d )
                    os << " ";
                os << positions( num_item, num_node, dim );
            }
        }
        os << ";";
        for( TI num_face = 0; num_face < nb_faces; ++num_face )
            os << " " << face_ids( num_item, num_face );
        os << "; " << ids( num_item );
    }
}

