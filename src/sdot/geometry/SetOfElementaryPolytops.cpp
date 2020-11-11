#include "SetOfElementaryPolytops.h"
#include <parex/support/ASSERT.h>
#include <parex/support/TODO.h>
#include <parex/support/P.h>

#include <parex/KernelCode.h>
#include <parex/Scheduler.h>

namespace sdot {

SetOfElementaryPolytops::SetOfElementaryPolytops( unsigned dim, std::string scalar_type, std::string index_type ) : scalar_type( scalar_type ), index_type( index_type ), dim( dim ) {
    parex::scheduler.kernel_code.add_include_dir( parex::KernelCode::path( SDOT_DIR ) / "src" / "sdot" / "kernels" );
    parex::scheduler.kernel_code.add_include_dir( parex::KernelCode::path( SDOT_DIR ) / "src" );

    parex::scheduler.kernel_code.includes[ "sdot::ShapeData" ].push_back( "<sdot/geometry/ShapeData.h>" );
    parex::scheduler.kernel_code.includes[ "sdot::ShapeMap" ].push_back( "<sdot/geometry/ShapeMap.h>" );

    shape_map = parex::Task::call_r( "SetOfElementaryPolytops/new_shape_map", {
        parex::Task::ref_type( scalar_type ),
        parex::Task::ref_type( index_type ),
        parex::Task::ref_num( dim ),
    } );
}

void SetOfElementaryPolytops::write_to_stream( std::ostream &os ) const {
    os << shape_map;
}

void SetOfElementaryPolytops::display_vtk( const std::string &filename ) const {
    parex::scheduler << parex::Task::call( parex::Kernel::with_priority( 10, "SetOfElementaryPolytops/display_vtk" ), {}, {
        shape_map.ref, parex::Task::ref_on( new std::string( filename ) )
    } );
}

void SetOfElementaryPolytops::add_repeated( ShapeType *shape_type, const Value &count, const Value &coordinates, const Value &face_ids, const Value &beg_ids ) {
    parex::Task::call( parex::Kernel::with_task_as_arg( "SetOfElementaryPolytops/add_repeated" ), { &shape_map.ref }, {
        shape_map.ref, parex::Task::ref_on( shape_type, false ), count.ref, coordinates.ref, face_ids.ref, beg_ids.ref
    } );
}

void SetOfElementaryPolytops::plane_cut( const Value &normals, const Value &scalar_products, const Value &/*cut_ids*/ ) {


    //    // get scalar product, cases and new_item_count
    //    std::vector<parex::TaskRef> rne = { parex::Task::ref_type( index_type ) };
    //    for( const auto &p : shape_map ) {
    //        const ShapeData &sd = p.second;

    //        parex::Task::call( new parex::Kernel{ "plane_cut_scalar_products" }, {
    //            &sd.cut_case_offsets.ref, &sd.indices.ref, &sd.scalar_products.ref, &sd.reservation_new_elements.ref
    //        }, {
    //            sd.coordinates.ref, sd.ids.ref, normals.ref, scalar_products.ref,
    //            parex::Task::ref_on( sd.shape_type->cut_poss_count(), false ),
    //            parex::Task::ref_on( sd.shape_type->cut_rese_new(), false ),
    //            parex::Task::ref_num( sd.shape_type->nb_nodes() ),
    //            parex::Task::ref_num( dim )
    //        } );

    //        rne.push_back( sd.reservation_new_elements.ref );
    //    }

    //    // make new shape data
    //    Value reservation_new_elements = parex::Task::call_r( new parex::Kernel{ "plane_cut_reservation_new_elements" }, std::move( rne ) );
    //    P( reservation_new_elements );
}

//        // update of nb items to create for each type
//        sd.shape_type->cut_rese( [&]( const ShapeType *shape_type, BI count ) {
//            auto iter = new_item_count.find( shape_type );
//            if ( iter == new_item_count.end() )
//                new_item_count.insert( iter, { shape_type, count } );
//            else
//                iter->second += count;
//        }, ks, sd );

//        int cpt = 0;
//        for( auto v : sd.cut_case_offsets )
//            P( cpt++, v );

//        // free local data
//        ks->free_TI( cut_cases );
//        ks->free_TI( offsets );
//    }

//    // new shape map (using the new item counts)
//    ShapeMap old_shape_map = std::exchange( shape_map, {} );
//    for( auto p : new_item_count )
//        shape_data( p.first )->reserve( p.second );

//    //
//    for( const auto &p : old_shape_map ) {
//        const ShapeData &sd = p.second;
//        sd.shape_type->cut_ops( ks, shape_map, sd, cut_ids.data(), dim );

//        // free tmp data from old shape map
//        ks->free_TF( sd.cut_out_scps );
//        ks->free_TI( sd.cut_indices  );
//    }
//}

}
