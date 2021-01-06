#include "HomogeneousElementaryPolytopList.h"

#define T template<class Allocator_TF,class Allocator_TI,int _nb_nodes,int _nb_faces,int _dim>
#define C HomogeneousElementaryPolytopList<Allocator_TF,Allocator_TI,_nb_nodes,_nb_faces,_dim>

T C::HomogeneousElementaryPolytopList( const Allocator_TF &allocator_TF, const Allocator_TI &allocator_TI, TI rese_items ) :
        positions( XP::empty( allocator_TF, 0, rese_items ) ),
        face_ids( XF::empty( allocator_TI, 0, rese_items ) ),
        ids( XI::empty( allocator_TI, 0, rese_items ) ) {
}

T void C::write_to_stream( std::ostream &os, const std::string &sp ) const {
    parex::get_local( positions.default_processor(), positions.data(), positions.nb_reserved_items(), [&]( auto position_data ) {
        parex::get_local( face_ids.default_processor(), face_ids.data(), face_ids.nb_reserved_items(), [&]( auto face_id_data ) {
            parex::get_local( ids.default_processor(), ids.data(), ids.nb_reserved_items(), [&]( auto id_data ) {
                for( TI num_item = 0; num_item < size(); ++num_item ) {
                    os << sp;
                    for( TI num_node = 0; num_node < nb_nodes; ++num_node ) {
                        if ( num_node )
                            os << ", ";
                        for( TI d = 0; d < dim; ++d ) {
                            if ( d )
                                os << " ";
                            os << position_data[ positions.offset( num_node, d, num_item ) ];
                        }
                    }
                    os << ";";
                    for( TI num_face = 0; num_face < nb_faces; ++num_face )
                        os << " " << face_id_data[ face_ids.offset( num_face, num_item ) ];
                    os << "; " << id_data[ ids.offset( num_item ) ];
                }
            } );
        } );
    } );
}

T template<class Proc> void C::resize( TI new_size, const Proc &proc ) {
    positions.resize( new_size, proc );
    face_ids.resize( new_size, proc );
    ids.resize( new_size, proc );
}

T typename C::TI C::size() const {
    return ids.size();
}

#undef T
#undef C
