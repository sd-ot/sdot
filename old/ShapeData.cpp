#include <parex/Kernel.h>
#include "ShapeType.h"
#include "ShapeData.h"
using namespace parex;

namespace sdot {

//ShapeData::ShapeData( const ShapeType *shape_type, std::size_t dim, std::string scalar_type, std::string index_type ) : shape_type( shape_type ) {
//    coordinates = Task::call_r( new Kernel{ "New_paren" }, { Task::ref_type( "parex::Tensor<" + scalar_type + ">" ), Task::ref_on( new Vec<std::size_t>{ 0, dim * shape_type->nb_nodes() } ) } );
//    face_ids    = Task::call_r( new Kernel{ "New_paren" }, { Task::ref_type( "parex::Tensor<" + index_type + ">" ), Task::ref_on( new Vec<std::size_t>{ 0, shape_type->nb_nodes() } ) } );
//    ids         = Task::call_r( new Kernel{ "New_paren" }, { Task::ref_type( "parex::Vec<" + index_type + ">" ) } );
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

} // namespace sdot
