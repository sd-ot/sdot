#include "SetOfElementaryPolytops.h"
#include <parex/support/P.h>
#include <parex/Scheduler.h>

namespace sdot {

SetOfElementaryPolytops::SetOfElementaryPolytops( const ElementaryPolytopTypes &ept, std::string scalar_type, std::string index_type ) : scalar_type( scalar_type ), index_type( index_type ), ept( ept ) {
    parex::scheduler.kernel_code.includes[ "sdot::ElementaryPolytopOperations" ].push_back( "<sdot/geometry/ElementaryPolytopOperations.h>" );
    parex::scheduler.kernel_code.includes[ "sdot::ShapeCutTmpData" ].push_back( "<sdot/geometry/ShapeCutTmpData.h>" );
    parex::scheduler.kernel_code.includes[ "sdot::ShapeType" ].push_back( "<sdot/geometry/ShapeType.h>" );
    parex::scheduler.kernel_code.includes[ "sdot::ShapeData" ].push_back( "<sdot/geometry/ShapeData.h>" );
    parex::scheduler.kernel_code.includes[ "sdot::ShapeMap" ].push_back( "<sdot/geometry/ShapeMap.h>" );

    parex::scheduler.kernel_code.add_include_dir( parex::KernelCode::path( SDOT_DIR ) / "src" );

    // void shape map
    shape_map = parex::Task::call_r( "sdot/geometry/kernels/SetOfElementaryPolytops/new_shape_map", {
        parex::Task::ref_type( scalar_type ),
        parex::Task::ref_type( index_type ),
        ept.dim,
    } );

}

void SetOfElementaryPolytops::write_to_stream( std::ostream &os ) const {
    os << shape_map;
}

void SetOfElementaryPolytops::display_vtk( const Value &filename ) const {
    parex::scheduler << parex::Task::call( parex::Kernel::with_priority( 10, "sdot/geometry/kernels/SetOfElementaryPolytops/display_vtk" ), {}, {
        filename.ref, shape_map.ref, ept.operations
    } );
}

void SetOfElementaryPolytops::add_repeated( const Value &shape_name, const Value &count, const Value &coordinates, const Value &face_ids, const Value &beg_ids ) {
    parex::Task::call( parex::Kernel::with_task_as_arg( "sdot/geometry/kernels/SetOfElementaryPolytops/add_repeated" ), { &shape_map.ref }, {
        shape_map.ref, shape_name.ref, ept.operations, count.ref, coordinates.ref, face_ids.ref, beg_ids.ref
    } );
}

void SetOfElementaryPolytops::plane_cut( const Value &normals, const Value &scalar_products, const Value &cut_ids ) {
    shape_map = parex::Task::call_r( parex::Kernel::with_task_as_arg( "sdot/geometry/kernels/SetOfElementaryPolytops/plane_cut" ), {
        shape_map.ref, normals.ref, scalar_products.ref, cut_ids.ref
    }, /*append_parent_task*/ true );
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
