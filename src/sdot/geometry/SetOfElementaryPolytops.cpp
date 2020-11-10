#include "SetOfElementaryPolytops.h"
#include <parex/support/ASSERT.h>
#include <parex/support/TODO.h>
#include <parex/support/P.h>
#include <parex/Scheduler.h>
#include <parex/KernelCode.h>

namespace sdot {

SetOfElementaryPolytops::SetOfElementaryPolytops( unsigned dim, std::string scalar_type, std::string index_type ) : scalar_type( scalar_type ), index_type( index_type ), dim( dim ) {
    parex::kernel_code.add_include_dir( parex::KernelCode::path( SDOT_DIR ) / "src" / "sdot" / "kernels" );
    // Prop: on démarre
}

void SetOfElementaryPolytops::write_to_stream( std::ostream &os, const std::string &sp ) const {
    os << sp << "SetOfElementaryPolytops([";
    Value vsp( sp + "  " );
    for( const auto &p : shape_map ) {
        const ShapeData &sd = p.second;
        os << "\n" << sp << " " << p.first->name() << "\n" << Value(
            parex::Task::call_r( new parex::Kernel{ "shape_data_to_string" }, {
                sd.coordinates.ref, sd.face_ids.ref, sd.ids.ref, vsp.ref
            }
        ) );
    }
    os << "\n" << sp << "])";
}

void SetOfElementaryPolytops::display_vtk( VtkOutput &vo, VtkOutput::Pt *offsets ) const {
    for( const auto &p : shape_map ) {
        const ShapeData &sd = p.second;
        p.first->display_vtk( [&]( Value vtk_id, Value nodes ) {
            parex::scheduler << parex::Task::call( new parex::Kernel{ "shape_data_display_vtk" }, {}, {
                parex::Task::ref_on( &vo, false ), parex::Task::ref_on( offsets, false ), sd.coordinates.ref, sd.face_ids.ref, sd.ids.ref,
                parex::Task::ref_num( dim ), vtk_id.ref, nodes.ref
            } );
            parex::scheduler.run();
        } );
    }
}

void SetOfElementaryPolytops::add_repeated( ShapeType *shape_type, const Value &count, const Value &coordinates, const Value &face_ids, const Value &beg_ids ) {
    ShapeData *sd = shape_data( shape_type );

    parex::Task::call( new parex::Kernel{ .name = "add_repeated_elements", .task_as_arg = true }, {
        &sd->coordinates.ref, &sd->face_ids.ref, &sd->ids.ref
    }, {
        sd->coordinates.ref, sd->face_ids.ref, sd->ids.ref,
        count.ref, coordinates.ref, face_ids.ref, beg_ids.ref
    } );
}

void SetOfElementaryPolytops::plane_cut( const Value &normals, const Value &scalar_products, const Value &/*cut_ids*/ ) {
    // get scalar product, cases and new_item_count
    for( const auto &p : shape_map ) {
        const ShapeData &sd = p.second;

        parex::Task::call( new parex::Kernel{ "plane_cut_scalar_products" }, {
            &sd.cut_case_offsets.ref, &sd.indices.ref, &sd.scalar_products.ref
        }, {
            sd.coordinates.ref, sd.ids.ref, normals.ref, scalar_products.ref,
            parex::Task::ref_on( sd.shape_type->cut_poss_count(), false ),
            parex::Task::ref_num( sd.shape_type->nb_nodes() ),
            parex::Task::ref_num( dim )
        } );

        P( sd.cut_case_offsets );
        P( sd.indices );
    }

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

ShapeData *SetOfElementaryPolytops::shape_data( const ShapeType *shape_type ) {
    auto iter = shape_map.find( shape_type );
    if ( iter == shape_map.end() )
        iter = shape_map.insert( iter, { shape_type, ShapeData{ shape_type, dim, scalar_type, index_type } } );
    return &iter->second;
}

}
