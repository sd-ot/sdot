#include <sstream>
#include <vector>
#include <string>
#include <array>
#include <set>
using TI = std::size_t;

void make_ElementaryPolytopOperations( std::ostream &os, const std::string &kernel_name, const std::string &parameter ) {
    std::vector<std::string> elem_names;
    std::istringstream is( parameter );
    for( std::string elem_name; is >> elem_name; )
        ;

    os << "#include <sdot/geometry/ElementaryPolytopOperations.h>\n";
    os << "#include <parex/TaskRef.h>\n";
    os << "using namespace parex;\n";
    os << "using namespace sdot;\n";
    os << "\n";
    os << "// " << parameter << "\n";

    os << "ElementaryPolytopOperations *" << kernel_name << "() {\n";
    os << "    ElementaryPolytopOperations *res = new ElementaryPolytopOperations;\n";
    for( std::string elem_name : elem_names ) {
        os << "    Operations &op = res->operation_map[ " << elem_name << " ];\n";
    }
    os << "    return res;\n";
    os << "}\n";
}
