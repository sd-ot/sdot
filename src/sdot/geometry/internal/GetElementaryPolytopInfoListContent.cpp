#include "GetElementaryPolytopInfoListContent.h"
#include <parex/GeneratedLibrarySet.h>
#include <parex/variable_encode.h>
#include <parex/CppType.h>
#include <parex/ERROR.h>
#include <parex/TODO.h>

namespace sdot {

GetElementaryPolytopInfoListContent::GetElementaryPolytopInfoListContent( const Rc<Task> &shape_types ) : ComputableTask( { shape_types } ) {
}

void GetElementaryPolytopInfoListContent::write_to_stream( std::ostream &os ) const {
    os << "GetElementaryPolytopInfoList";
}

void GetElementaryPolytopInfoListContent::exec() {
    // signature / output type name
    std::string shape_types = *reinterpret_cast<std::string *>( children[ 0 ]->output_data );
    if ( shape_types.empty() )
        ERROR( "" );
    if ( shape_types[ 0 ] != '[' )
        shape_types = default_shape_types( std::stoi( shape_types ) );
    if ( shape_types.back() != ']' )
        ERROR( "" );
    shape_types = shape_types.substr( 1, shape_types.size() - 2 );

    std::string sg = "ElementaryPolytopInfoListContent_" + variable_encode( shape_types );

    // create or get the type
    Type *output_type = type_factory().reg_cpp_type( sg, [&]( CppType &ct ) {
        // includes
        ct.includes << "<sdot/geometry/internal/ElementaryPolytopInfoListContent.h>";
        ct.include_directories << SDOT_DIR "/src";

        // preliminaries
        std::ostringstream sd;
        sd << "\n";
        sd << "struct " << sg << " : ElementaryPolytopInfoListContent {\n";
        sd << "    /***/        " << sg << "() {\n";
        write_ctor( sd, std::istringstream{ shape_types }, "\n        " );
        sd << "    }\n";
        sd << "    virtual void write_to_stream( std::ostream &os ) const {\n";
        sd << "        os << \"" << sg << "\";\n";
        sd << "    }\n";
        sd << "};\n";
        sd << "\n";
        sd << "inline std::string type_name( S<" << sg << "> ) { return \"" << sg << "\"; };\n";
        ct.preliminaries << sd.str();

    } );

    // find or create lib
    static GeneratedLibrarySet gls;
    DynamicLibrary *lib = gls.get_library( [&]( SrcSet &sw ) {
        Src &src = sw.src( sg + ".cpp" );

        src.includes << "<parex/ComputableTask.h>";
        output_type->add_needs_in( src );

        src << "extern \"C\" void get_ptr_GetElementaryPolytopInfoListContent( ComputableTask *task ) {\n";
        src << "    static " << sg << " output_data;\n";
        src << "    \n";
        src << "    task->output_type = task->type_factory_virtual( \"" << output_type->cpp_name() << "\");\n";
        src << "    task->output_data = &output_data;\n";
        src << "    task->output_own = false;\n";
        src << "}\n";
    }, sg );

    // execute the generated function
    auto *func = lib->symbol<void( ComputableTask *)>( "get_ptr_GetElementaryPolytopInfoListContent" );
    func( this );
}

std::string GetElementaryPolytopInfoListContent::default_shape_types( int dim ) {
    if ( dim == 3 ) return "[3S 3E 4S]";
    if ( dim == 2 ) return "[3 4 5]";
    TODO;
    return {};
}

void GetElementaryPolytopInfoListContent::write_ctor( std::ostream &os, std::istringstream &&shape_types, const std::string &sp ) {

}

} // namespace sdot
