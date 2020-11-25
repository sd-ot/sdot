#include "SetOfElementaryPolytops.h"
#include <parex/support/TODO.h>
#include <parex/support/P.h>
#include <parex/Scheduler.h>
using namespace parex;

namespace sdot {

SetOfElementaryPolytops::SetOfElementaryPolytops( const TaskRef &n_dim, const TaskRef &s_scalar_type, const TaskRef &s_index_type, std::string shape_list ) : shape_list( shape_list ) {
    // types
    std::string ds = "sdot/geometry/kernels/SetOfElementaryPolytops/data_structures/";

    scheduler.kernel_code.includes[ "sdot::ShapeCutTmpData" ].push_back( "<" + ds + "ShapeCutTmpData.h>" );
    scheduler.kernel_code.includes[ "sdot::ShapeData" ].push_back( "<" + ds + "ShapeData.h>" );
    scheduler.kernel_code.includes[ "sdot::ShapeMap" ].push_back( "<" + ds + "ShapeMap.h>" );

    // includes
    scheduler.kernel_code.add_include_dir( KernelCode::path( SDOT_DIR ) / "src" );

    // creation of a void shape map
    shape_map = Task::call_r( "sdot/geometry/kernels/SetOfElementaryPolytops/new_shape_map", {
        s_scalar_type,
        s_index_type,
        n_dim
    } );
}

SetOfElementaryPolytops::SetOfElementaryPolytops( int dim, std::string scalar_type, std::string index_type ) :
        SetOfElementaryPolytops( Task::ref_type( scalar_type ), Task::ref_type( index_type ), Task::ref_num( dim ), default_shape_list( dim ) ) {
}

void SetOfElementaryPolytops::write_to_stream( std::ostream &os ) const {
    os << shape_map;
}

void SetOfElementaryPolytops::display_vtk( const Value &filename ) const {
    scheduler << Task::call( Kernel::with_priority( 10, "sdot/geometry/kernels/SetOfElementaryPolytops/display_vtk" ), {}, {
        filename.ref, shape_map.ref
    } );
}

void SetOfElementaryPolytops::add_repeated( const Value &shape_name, const Value &count, const Value &coordinates, const Value &face_ids, const Value &beg_ids ) {
    Task::call( Kernel::with_task_as_arg( "sdot/geometry/kernels/SetOfElementaryPolytops/add_repeated" ), { &shape_map.ref }, {
        shape_map.ref, shape_name.ref, ept.operations, count.ref, coordinates.ref, face_ids.ref, beg_ids.ref
    } );
}

void SetOfElementaryPolytops::plane_cut( const Value &normals, const Value &scalar_products, const Value &cut_ids ) {
    shape_map = Task::call_r( Kernel::with_task_as_arg( "sdot/geometry/kernels/SetOfElementaryPolytops/plane_cut" ), {
        shape_map.ref, ept.operations, normals.ref, scalar_products.ref, cut_ids.ref
                              }, /*append_parent_task*/ true );
}

std::string SetOfElementaryPolytops::default_shape_list( int dim ) {
    if ( dim == 3 ) return "3S 3E 4S";
    if ( dim == 2 ) return "3 4 5";
    TODO;
    return {};
}

}
