#include <parex/GeneratedLibrarySet.h>
#include <parex/variable_encode.h>
#include <parex/CompiledTask.h>
#include <parex/Scheduler.h>
#include <parex/CppType.h>
#include <parex/TODO.h>
#include <parex/P.h>
#include <sstream>

#include "internal/ElementaryPolytopInfoListContent.h"
#include "SetOfElementaryPolytops.h"

namespace sdot {

SetOfElementaryPolytops::SetOfElementaryPolytops( const ElementaryPolytopInfoList &elementary_polytop_info, const Value &scalar_type, const Value &index_type, const Value &dim ) {
    struct GetShapeMap : ComputableTask {
        using ComputableTask::ComputableTask;

        virtual void write_to_stream( std::ostream &os ) const override {
            os << "GetShapeMap";
        }

        virtual void exec() override {
            // inputs
            const ElementaryPolytopInfoListContent *epil = reinterpret_cast<const ElementaryPolytopInfoListContent *>( children[ 0 ]->output_data );
            Type *scalar_type = Task::type_factory( *reinterpret_cast<const std::string *>( children[ 1 ]->output_data ) );
            Type *index_type = Task::type_factory( *reinterpret_cast<const std::string *>( children[ 2 ]->output_data ) );
            int dim = *reinterpret_cast<const int *>( children[ 3 ]->output_data );
            if ( ! dim )
                dim = epil->default_dim;

            // type name
            std::string type_name = "ShapeMap";
            type_name += "_" + variable_encode( epil->elem_names(), true );
            type_name += "_" + variable_encode( scalar_type->cpp_name(), true );
            type_name += "_" + variable_encode( index_type->cpp_name(), true );
            type_name += "_" + std::to_string( dim );

            P( type_name );
            TODO;

            //            // output type
            //            output_type = type_factory().reg_cpp_type( "ElementaryPolytopInfoListContent", []( CppType &ct ) {
            //                ct.includes << "<sdot/geometry/internal/ElementaryPolytopInfoListContent.h>";
            //                ct.include_directories << SDOT_DIR "/src";
            //            } );

            //            // summary
            //            std::string shape_types = *reinterpret_cast<const std::string *>( children[ 0 ]->output_data );

            //            // find or create lib
            //            static GeneratedLibrarySet gls;
            //            DynamicLibrary *lib = gls.get_library( [&]( SrcSet &sw ) {
            //                Src &src = sw.src( "get_ElementaryPolytopInfoList.cpp" );
            //                src.includes << "<parex/ComputableTask.h>";
            //                output_type->add_needs_in( src );

            //                src << "\n";
            //                src << "extern \"C\" void kernel( ComputableTask *task ) {\n";
            //                src << "    static Epil res;\n";
            //                src << "    task->output_data = &res;\n";
            //                src << "    task->output_own = false;\n";
            //                src << "}\n";
            //            }, shape_types );

            //            // execute the generated function to get the output_data
            //            auto *func = lib->symbol<void( ComputableTask *)>( "kernel" );
            //            func( this );
        }
    };

    shape_map = new GetShapeMap( {
        elementary_polytop_info.task,
        scalar_type.to_string(),
        index_type.to_string(),
        dim.conv_to<int>()
    } );
}

void SetOfElementaryPolytops::write_to_stream( std::ostream &os ) const {
    os << Value( shape_map );
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
