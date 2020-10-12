#include "ShapeData.h"

namespace sdot {

ShapeData::ShapeData( KernelSlot *ks, ShapeType *shape_type, unsigned dim ) : shape_type( shape_type ), nb_items( 0 ), ids( ks ) {
    coordinates.resize( shape_type->nb_nodes() * dim, ks );
    face_ids.resize( shape_type->nb_faces(), ks );
}

void ShapeData::resize( BI new_size ) {
    if ( coordinates.size() && coordinates[ 0 ].size() < new_size )
        for( VecTF &c : coordinates )
            c.resize( new_size );

    nb_items = new_size;
}

ShapeData::BI ShapeData::size() const {
    return nb_items;
}

}
