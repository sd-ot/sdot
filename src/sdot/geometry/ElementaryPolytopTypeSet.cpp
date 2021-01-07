#include <parex/instructions/CompiledLambdaInstruction.h>
#include "internal/GetElementaryPolytopCaracList.h"
#include "ElementaryPolytopTypeSet.h"

namespace sdot {

ElementaryPolytopTypeSet::ElementaryPolytopTypeSet( const parex::Vector<parex::String> &shape_names ) {
    carac = new parex::Variable( new GetElementaryPolytopCaracList( shape_names ), 1 );
}

ElementaryPolytopTypeSet::ElementaryPolytopTypeSet( const parex::Number &dim ) : ElementaryPolytopTypeSet( default_shape_names_for( dim ) ) {
}

parex::Vector<parex::String> ElementaryPolytopTypeSet::default_shape_names_for( const parex::Number &dim ) {
    return { new parex::CompiledLambdaInstruction( "GetDefaultShapes", { dim.variable->get() }, []( parex::Src &src, parex::SrcSet &, parex::TypeFactory * ) {
        src.compilation_environment.includes << "<parex/utility/TODO.h>";
        src.compilation_environment.includes << "<vector>";
        src.compilation_environment.includes << "<string>";

        src << "std::vector<std::string> *func( int dim ) {\n";
        src << "    if ( dim == 2 ) return new std::vector<std::string>{ \"3\" };\n";
        src << "    if ( dim == 3 ) return new std::vector<std::string>{ \"3S\" };\n";
        src << "    TODO; return nullptr;\n";
        src << "}\n";
        }, 1 ), 1 };
}

void ElementaryPolytopTypeSet::write_to_stream( std::ostream &os ) const {
    carac->display_data( os );
}

} // namespace sdot
