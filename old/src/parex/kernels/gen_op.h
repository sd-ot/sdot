#include <ostream>
#include <string>

void gen_op( std::ostream &os, const std::string &kernel_name, const std::string &parameter ) {
    os << "#include <parex/TaskRef.h>\n";
    os << "using namespace parex;\n";
    os << "\n";
    os << "template<class T>\n";
    os << "T *" << kernel_name << "( Task *t, T &a, T &b ) {\n";
    os << "    if ( t->move_arg( 0 ) )\n";
    os << "        return a " << parameter << "= b, nullptr;\n";
    os << "    if ( t->move_arg( 1 ) )\n";
    os << "        return b " << parameter << "= a, nullptr;\n";
    os << "    return new T( a " << parameter << " b );\n";
    os << "}\n";
}
