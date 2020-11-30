#include <parex/CompiledIncludeTask.h>
#include <parex/CompiledLambdaTask.h>
#include <parex/variable_encode.h>
#include <parex/Scheduler.h>
#include <parex/CppType.h>
#include <parex/TODO.h>
#include <parex/P.h>
#include <sstream>

#include "SetOfElementaryPolytops.h"

namespace sdot {

SetOfElementaryPolytops::SetOfElementaryPolytops( const ElementaryPolytopInfoList &elementary_polytop_info, const Value &scalar_type, const Value &index_type, const Value &dim ) {
    //    shape_map = static_cast<Task *>( new CompiledLambdaTask( []( Src &src, SrcSet &/*sw*/, const std::vector<Rc<Task>> &children ) {
    //        //        // get output type
    //        //        Type *map_type = type_of_shape_map(
    //        //            *reinterpret_cast<int *>( children[ 0 ]->output_data ),
    //        //            Task::type_factory( *reinterpret_cast<std::string *>( children[ 1 ]->output_data ) ),
    //        //            Task::type_factory( *reinterpret_cast<std::string *>( children[ 2 ]->output_data ) ),
    //        //            *reinterpret_cast<std::string *>( children[ 3 ]->output_data )
    //        //            );

    //        //        // register
    //        //        map_type->add_needs_in( src );

    //        //        // write src
    //        //        src << "template<class... A>\n";
    //        //        src << "TaskOut<" << map_type->cpp_name() << "> new_shape_map( const A& ... ) {\n";
    //        //        src << "    return new " << map_type->cpp_name() << ";\n";
    //        //        src << "}\n";
    //    }, { elementary_polytop_info.task, scalar_type.to_string(), index_type.to_string(), dim.conv_to<int>() }, "new_shape_map" ) );
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

//Type *SetOfElementaryPolytops::type_of_shape_map( int dim, Type *scalar_type, Type *index_type, std::string shape_types ) {
//    // shape types
//    if ( shape_types.empty() )
//        shape_types = default_shape_list( dim );

//    // output type name
//    std::ostringstream nt;
//    nt << "ShapeMap" << '_' << dim
//       << '_' << variable_encode( scalar_type->cpp_name(), true )
//       << '_' << variable_encode( index_type ->cpp_name(), true )
//       << '_' << variable_encode( shape_types, true );

//    // output type code
//    TypeFactory &tf = Task::type_factory();
//    return tf.reg_cpp_type( nt.str(), [&]( CppType &ct ) {
//        ct.includes << "<sdot/geometry/SetOfElementaryPolytops/HomogeneousElementaryPolytopList.h>";
//        ct.include_directories << SDOT_DIR "/ext/xtensor/install/include";
//        ct.include_directories << SDOT_DIR "/ext/xsimd/install/include";
//        ct.include_directories << SDOT_DIR "/src";

//        ct.preliminaries << "using namespace sdot;";

//        ct.sub_types.push_back( scalar_type );
//        ct.sub_types.push_back( index_type );

//        //
//        std::vector<ElementaryPolytopInfo> epil = elementary_polytop_info_list( std::istringstream{ shape_types } );

//        //
//        std::ostringstream sd;
//        sd << "\n";
//        sd << "struct " << nt.str() << " {\n";
//        sd << "    void write_to_stream( std::ostream &os ) const {\n";
//        sd << "        os << \"" << nt.str() << "\";\n";
//        sd << "    }\n";
//        sd << "    \n";
//        for( const ElementaryPolytopInfo &epi : epil )
//            sd << "    HomogeneousElementaryPolytopList<" << scalar_type->cpp_name() << "," << index_type->cpp_name() << "," << epi.nb_nodes() << "," << dim << "> sl_" << epi.name << ";\n";

//        sd << "};\n";
//        sd << "\n";
//        sd << "inline std::string type_name( S<" << nt.str() << "> ) { return \"" << nt.str() << "\"; };\n";
//        ct.preliminaries << sd.str();
//    } );
//}

} // namespace sdot
