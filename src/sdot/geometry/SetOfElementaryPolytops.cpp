#include <parex/CompiledIncludeTask.h>
#include <parex/CompiledLambdaTask.h>
#include <parex/variable_encode.h>
#include <parex/Scheduler.h>
#include <parex/CppType.h>
#include <parex/TODO.h>
#include <parex/P.h>

#include "SetOfElementaryPolytops.h"

namespace sdot {

SetOfElementaryPolytops::SetOfElementaryPolytops( Value dim, Value scalar_type, Value index_type, Value elem_shapes ) {
    shape_map = static_cast<Task *>( new CompiledLambdaTask( []( Src &src, SrcSet &/*sw*/, const std::vector<Rc<Task>> &children ) {
        // parameter values
        int dim = *reinterpret_cast<int *>( children[ 0 ]->output_data );
        std::string scalar_type = *reinterpret_cast<std::string *>( children[ 1 ]->output_data );
        std::string index_type  = *reinterpret_cast<std::string *>( children[ 2 ]->output_data );
        std::string shape_types = *reinterpret_cast<std::string *>( children[ 3 ]->output_data );
        if ( shape_types.empty() )
            shape_types = default_shape_list( dim );

        // make the output type
        std::ostringstream nt;
        nt << "ShapeMap" << '_' << dim
           << '_' << variable_encode( scalar_type, true )
           << '_' << variable_encode( index_type , true )
           << '_' << variable_encode( shape_types, true );

        TypeFactory &tf = Task::type_factory();
        Type *ns = tf.reg_cpp_type( nt.str(), [&]( CppType &ct ) {
            std::ostringstream sd;
            sd << "struct " << nt.str() << "{\n";
            sd << "};\n";
            ct.preliminaries.push_back( sd.str() );
        } );

        // write src
        ns->add_needs_in( src );
        src << "template<class... A>\n";
        src << "TaskOut<" << ns->cpp_name() << "> new_shape_map( const A& ... ) {\n";
        src << "    return new " << ns->cpp_name() << ";\n";
        src << "}\n";
    }, { dim.conv_to<int>(), scalar_type.to_string(), index_type.to_string(), elem_shapes.to_string() }, "new_shape_map" ) );

    //    // types
    //    std::string ds = "sdot/geometry/kernels/SetOfElementaryPolytops/data_structures/";

    //    scheduler.kernel_code.includes[ "sdot::ShapeCutTmpData" ].push_back( "<" + ds + "ShapeCutTmpData.h>" );
    //    scheduler.kernel_code.includes[ "sdot::ShapeData" ].push_back( "<" + ds + "ShapeData.h>" );
    //    scheduler.kernel_code.includes[ "sdot::ShapeMap" ].push_back( "<" + ds + "ShapeMap.h>" );

    //    // includes
    //    scheduler.kernel_code.add_include_dir( KernelCode::path( SDOT_DIR ) / "src" );

    //    // creation of a void shape map
    //    shape_map = Task::call_r( "sdot/geometry/kernels/SetOfElementaryPolytops/new_shape_map", {
    //        s_scalar_type,
    //        s_index_type,
    //        n_dim
    //    } );
}

void SetOfElementaryPolytops::write_to_stream( std::ostream &os ) const {
    os << shape_map;
}

//void SetOfElementaryPolytops::display_vtk( const Value &filename ) const {
//    scheduler << Task::call( Kernel::with_priority( 10, "sdot/geometry/kernels/SetOfElementaryPolytops/display_vtk" ), {}, {
//        filename.ref, shape_map.ref
//    } );
//}

//void SetOfElementaryPolytops::add_repeated( const Value &shape_name, const Value &count, const Value &coordinates, const Value &face_ids, const Value &beg_ids ) {
//    Task::call( Kernel::with_task_as_arg( "sdot/geometry/kernels/SetOfElementaryPolytops/add_repeated" ), { &shape_map.ref }, {
//        shape_map.ref, shape_name.ref, ept.operations, count.ref, coordinates.ref, face_ids.ref, beg_ids.ref
//    } );
//}

//void SetOfElementaryPolytops::plane_cut( const Value &normals, const Value &scalar_products, const Value &cut_ids ) {
//    shape_map = Task::call_r( Kernel::with_task_as_arg( "sdot/geometry/kernels/SetOfElementaryPolytops/plane_cut" ), {
//        shape_map.ref, ept.operations, normals.ref, scalar_products.ref, cut_ids.ref
//                              }, /*append_parent_task*/ true );
//}

std::string SetOfElementaryPolytops::default_shape_list( int dim ) {
    if ( dim == 3 ) return "3S 3E 4S";
    if ( dim == 2 ) return "3 4 5";
    TODO;
    return {};
}

}
