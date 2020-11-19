#include <sstream>
#include <vector>
#include <string>
#include <array>
#include <set>
using TI = std::size_t;

//std::vector<CutOp>                         new_element_creation_cut; ///< creation of new elements
//std::map<std::string,std::vector<TI>>      new_output_elements_cut;  ///< [output_elem_type][case_and_sub_case_number] => nb output elements
//std::vector<TI>                            nb_sub_cases_cut;         ///< [case_number] => nb_sub_cases
//std::vector<std::pair<TI,std::vector<TI>>> vtk_elements;             ///< [vtk_id + [node numbers]]

struct Elem {
    Elem( std::string name ) {
        nb_nodes = 0;
        std::size_t p = 0;
        for( ; p < name.size() && std::isdigit( name[ p ] ); ++p )
            nb_nodes += nb_nodes * 10 + ( name[ p ] - '0' );

        nb_faces = nb_nodes;
    }

    int nb_nodes;
    int nb_faces;
    int nvi;
};

void write_info_elem( std::ostream &os, std::string elem_name, std::string var_name, std::vector<std::string> elem_names ) {
    Elem elem( elem_name );
    os << "    " << var_name << ".nb_nodes = " << elem.nb_nodes << ";\n";
    os << "    " << var_name << ".nb_faces = " << elem.nb_faces << ";\n";
}

void make_ElementaryPolytopOperations( std::ostream &os, const std::string &kernel_name, const std::string &parameter ) {
    std::vector<std::string> elem_names;
    std::istringstream is( parameter );
    for( std::string elem_name; is >> elem_name; )
        elem_names.push_back( elem_name );

    os << "#include <sdot/geometry/ElementaryPolytopOperations.h>\n";
    os << "#include <parex/support/P.h>\n";
    os << "#include <parex/TaskRef.h>\n";
    os << "using namespace parex;\n";
    os << "using namespace sdot;\n";
    os << "\n";
    os << "// " << parameter << "\n";

    os << "ElementaryPolytopOperations *" << kernel_name << "() {\n";
    os << "    ElementaryPolytopOperations *res = new ElementaryPolytopOperations;\n";
    for( TI i = 0; i < elem_names.size(); ++i ) {
        std::string elem_name = elem_names[ i ];
        os << "    ElementaryPolytopOperations::Operations &op_" << i << " = res->operation_map[ \"" << elem_name << "\" ];\n";
        write_info_elem( os, elem_name, "op_" + std::to_string( i ), elem_names );
    }
    os << "    return res;\n";
    os << "}\n";
}
