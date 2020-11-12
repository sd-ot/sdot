#include "ShapeData.h"
#include "ShapeType.h"

namespace sdot {

template<class TF,class TI,int dim>
ShapeData<TF,TI,dim>::ShapeData( ShapeType *shape_type, TI nb_items ) :
    coordinates( parex::Vec<TI>{ nb_items, shape_type->nb_nodes() * dim } ),
    face_ids( parex::Vec<TI>{ nb_items, shape_type->nb_faces() } ),
    ids( nb_items ) {
}

template<class TF,class TI,int dim>
void ShapeData<TF,TI,dim>::write_to_stream( std::ostream &os, const std::string &sp ) const {
    for( TI i = 0; i < ids.size(); ++i ) {
        os << "\n" << sp;

        os << "C:";
        for( TI c = 0; c < coordinates.nb_x_vec(); ++c )
            os << " " << std::setw( 8 ) << coordinates.ptr( c )[ i ];

        os << " F:";
        for( TI c = 0; c < face_ids.nb_x_vec(); ++c )
            os << " " << std::setw( 6 ) << face_ids.ptr( c )[ i ];

        os << " I: " << ids[ i ];
    }
}

} // namespace sdot
