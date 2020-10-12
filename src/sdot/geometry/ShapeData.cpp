#include "ShapeData.h"

namespace sdot {

ShapeData::ShapeData( KernelSlot *ks, ShapeType *shape_type, unsigned dim ) : shape_type( shape_type ), ids( ks ) {
    coordinates.resize( shape_type->nb_nodes() * dim, ks );
    face_ids.resize( shape_type->nb_faces(), ks );
}

void ShapeData::resize( BI new_size ) {
    for( VecTF &c : coordinates )
        c.resize( new_size );
    for( VecTI &c : face_ids )
        c.resize( new_size );
    ids.resize( new_size );
}

ShapeData::BI ShapeData::size() const {
    return ids.size();
}

}
