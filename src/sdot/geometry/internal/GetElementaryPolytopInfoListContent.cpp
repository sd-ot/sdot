#include "GetElementaryPolytopInfoList.h"
#include <parex/GeneratedLibrarySet.h>
#include <parex/variable_encode.h>
#include <parex/CppType.h>
#include <parex/ERROR.h>
#include <parex/TODO.h>

namespace sdot {

GetElementaryPolytopInfoList::GetElementaryPolytopInfoList( const Rc<Task> &shape_types ) : ComputableTask( { shape_types } ) {
}

void GetElementaryPolytopInfoList::write_to_stream( std::ostream &os ) const {
    os << "GetElementaryPolytopInfoList";
}

void GetElementaryPolytopInfoList::exec() {
    // signature / output type name
    std::string shape_types = *reinterpret_cast<std::string *>( children[ 0 ]->output_data );
    if ( shape_types.empty() )
        ERROR( "" );
    if ( shape_types[ 0 ] >= '0' && shape_types[ 0 ] <= '9' )
        shape_types = default_shape_types( std::stoi( shape_types ) );

    std::string sg = "ElementaryPolytopInfoList_" + variable_encode( shape_types );

    // create or get the type
    Type *output_type = type_factory().reg_cpp_type( sg, [&]( CppType &ct ) {
        //
        // std::vector<ElementaryPolytopInfo> epil = elementary_polytop_info_list( std::istringstream{ shape_types } );

        //
        std::ostringstream sd;
        sd << "\n";
        sd << "struct " << sg << " {\n";
        sd << "    void write_to_stream( std::ostream &os ) const {\n";
        sd << "        os << \"" << sg << "\";\n";
        sd << "    }\n";
        sd << "    \n";
        //        for( const ElementaryPolytopInfo &epi : epil )
        //            sd << "    HomogeneousElementaryPolytopList<" << scalar_type->cpp_name() << "," << index_type->cpp_name() << "," << epi.nb_nodes() << "," << dim << "> sl_" << epi.name << ";\n";

        sd << "};\n";
        sd << "\n";
        sd << "inline std::string type_name( S<" << sg << "> ) { return \"" << sg << "\"; };\n";
        ct.preliminaries << sd.str();
    } );

    // find or create lib
    static GeneratedLibrarySet gls;
    DynamicLibrary *lib = gls.get_library( [&]( SrcSet &sw ) {
        Src &src = sw.src( sg + ".cpp" );

        src.include_directories << PAREX_DIR "/src";
        src.includes << "<parex/ComputableTask.h>";

        src << "extern \"C\" void get_GetElementaryPolytopInfoList( ComputableTask *task ) {\n";
        src << "    task->output_type = task->type_factory_virtual( \"" << output_type->cpp_name() << "\");\n";
        src << "    task->output_data = nullptr;\n";
        src << "    task->output_own = false;\n";
        src << "}\n";
    }, sg );

    // execute the generated function
    auto *func = lib->symbol<void( ComputableTask *)>( "get_GetElementaryPolytopInfoList" );
    func( this );

}

std::string GetElementaryPolytopInfoList::default_shape_types( int dim ) {
    if ( dim == 3 ) return "CH3S CH3E CH4S";
    if ( dim == 2 ) return "CH3 CH4 CH5";
    TODO;
    return {};
}

} // namespace sdot
