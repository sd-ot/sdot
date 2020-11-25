#include "data_structures/generation/Element.h"
#include <sstream>

void write_info_elem( std::ostream &os, Element &element, std::string var_name, std::map<std::string,Element> &elements ) {
    os << "    " << var_name << ".nb_nodes = " << element.nb_nodes << ";\n";
    os << "    " << var_name << ".nb_faces = " << element.nb_faces << ";\n";
    element.write_cut_info( os, var_name, elements );
    element.write_vtk_info( os, var_name );
}

void make_ElementaryPolytopOperations( std::ostream &os, const std::string &kernel_name, const std::string &parameter ) {
    std::map<std::string,Element> elements;
    std::istringstream is( parameter );
    for( std::string elem_name; is >> elem_name; )
        elements.insert( { elem_name, { elem_name } } );

    os << "#include <sdot/geometry/kernels/SetOfElementaryPolytops/data_structures/ElementaryPolytopOperations.h>\n";
    os << "#include <parex/support/P.h>\n";
    os << "#include <parex/TaskRef.h>\n";
    os << "using namespace parex;\n";
    os << "using namespace sdot;\n";
    os << "\n";
    os << "// " << parameter << "\n";

    os << "ElementaryPolytopOperations *" << kernel_name << "() {\n";
    os << "    ElementaryPolytopOperations *res = new ElementaryPolytopOperations;\n";
    for( auto &p : elements ) {
        os << "\n    ElementaryPolytopOperations::Operations &op_" << p.first << " = res->operation_map[ \"" << p.first << "\" ];\n";
        write_info_elem( os, p.second, "op_" + p.first, elements );
    }
    os << "    return res;\n";
    os << "}\n";
}
