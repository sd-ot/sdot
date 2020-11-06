#include "ShapeData.h"

namespace parex {

//ShapeData::ShapeData( KernelSlot *ks, const ShapeType *shape_type, unsigned dim ) : shape_type( shape_type ), dim( dim ), ks( ks ), coordinates( ks ), face_ids( ks ), ids( ks ) {
//    log2_rese = 0;
//    rese      = 0;
//    size      = 0;
//}

//ShapeData::~ShapeData() {
//    if ( rese ) {
//        ks->free_TF( coordinates );
//        ks->free_TI( face_ids );
//        ks->free_TI( ids );
//    }
//}

//void ShapeData::reserve( BI new_rese ) {
//    if ( rese >= new_rese )
//        return;

//    // old values
//    void *old_coordinates = coordinates;
//    void *old_face_ids = face_ids;
//    void *old_ids = ids;
//    BI old_rese = rese;

//    // update rese
//    if ( ! rese ) {
//        log2_rese = 5;
//        rese = 32; // for base alignement
//    }
//    while ( rese < new_rese ) {
//        log2_rese += 1;
//        rese *= 2;
//    }

//    // allocate
//    coordinates = ks->allocate_TF( shape_type->nb_nodes() * dim * rese );
//    face_ids = ks->allocate_TI( shape_type->nb_faces() * rese );
//    ids = ks->allocate_TI( rese );

//    // copy old data
//    if ( size ) {
//        for( unsigned i = 0; i < shape_type->nb_nodes() * dim; ++i )
//            ks->assign_TF( coordinates, i * rese, old_coordinates, i * old_rese, size );
//        for( unsigned i = 0; i < shape_type->nb_faces(); ++i )
//            ks->assign_TI( face_ids, i * rese, old_face_ids, i * old_rese, size );
//        ks->assign_TI( ids, 0, old_ids, 0, size );
//    }

//    // free old data
//    if ( old_rese ) {
//        ks->free_TF( old_coordinates );
//        ks->free_TI( old_face_ids );
//        ks->free_TI( old_ids );
//    }
//}

//void ShapeData::resize( BI new_size ) {
//    reserve( new_size );
//    size = new_size;
//}

}
