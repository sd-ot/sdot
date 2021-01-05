#include "HomogeneousElementaryPolytopList.h"

#define T template<class Allocator_TF,class Allocator_TI,int nb_nodes,int nb_faces,int dim>
#define C HomogeneousElementaryPolytopList<Allocator_TF,Allocator_TI,nb_nodes,nb_faces,dim>

T C::HomogeneousElementaryPolytopList( const Allocator_TF &allocator_TF, const Allocator_TI &allocator_TI, TI rese_items ) :
        positions( XP::empty( allocator_TF, 0, rese_items ) ),
        face_ids( XF::empty( allocator_TI, 0, rese_items ) ),
        ids( XI::empty( allocator_TI, 0, rese_items ) ) {
}

T void C::write_to_stream( std::ostream &os, const std::string &sp ) const {
    for( TI num_item = 0; num_item < size(); ++num_item ) {
        os << sp;
        for( TI num_node = 0; num_node < nb_nodes; ++num_node ) {
            if ( num_node )
                os << ", ";
            for( TI d = 0; d < dim; ++d ) {
                if ( d )
                    os << " ";
                //os << positions.at( af, num_node, d, num_item );
            }
        }
        os << ";";
        //        for( TI num_face = 0; num_face < nb_faces(); ++num_face )
        //            os << " " << face_ids.at( ai, num_face, num_item );
        //        os << "; " << ids.at( ai, num_item );
    }
}

T template<class Proc> void C::resize( TI new_size, const Proc &proc ) {
    positions.resize_axis( 2, new_size, proc );
    face_ids.resize_axis( 1, new_size, proc );
    ids.resize_axis( 0, new_size, proc );
}

T typename C::TI C::size() const {
    return ids.size();
}

#undef T
#undef C
