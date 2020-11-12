#include <sstream>
#include <vector>
#include <string>
#include <array>

struct ParmReader {
    struct Output {
        std::vector<std::array<int,2>> inp_nodes;
        std::vector<int> inp_faces;
    };

    ParmReader( std::string parm ) {
        // 3 0 0 1 1 2 2 3 0 1 2
        std::istringstream ss( parm );
        int n, a, b;
        while( true ) {
            // nb nodes
            ss >> n;
            if ( ! ss )
                break;

            // nodes
            Output output;
            while ( n-- ) {
                ss >> a >> b;
                output.inp_nodes.push_back( { a, b } );
            }

            // faces
            ss >> n;
            while ( n-- ) {
                ss >> a;
                output.inp_faces.push_back( a );
            }

            outputs.push_back( output );
        }
    }

    std::vector<Output> outputs;
};

void mk_items( std::ostream &os, const std::string &kernel_name, const std::string &parameter ) {
    ParmReader pr( parameter );

    os << "#include <sdot/geometry/ShapeMap.h>\n";
    os << "#include <parex/support/P.h>\n";
    os << "#include <parex/TaskRef.h>\n";
    os << "using namespace parex;\n";
    os << "using namespace sdot;\n";
    os << "\n";
    os << "// " << parameter << "\n";

    os << "template<class TF,class TI,int dim,class VI>\n";
    os << "ShapeMap<TF,TI,dim> *" << kernel_name << "( Task *t, ShapeMap<TF,TI,dim> &sm";
    for( std::size_t num_output = 0; num_output < pr.outputs.size(); ++num_output )
        os << ", ShapeType &out_shape_type_" << num_output
           << ", const VI &out_node_corr_" << num_output
           << ", const VI &out_face_corr_" << num_output << "\n";
    os << ", const VI &inp_node_corr, const VI &inp_face_corr, TI beg, TI end ) {\n";
    os << "    P( \"" << parameter << "\" );\n";
    os << "    P( t->children[ 0 ].task->kernel.name );\n";
    os << "    P( t->children[ 0 ].task->ref_count );\n";
    os << "    P( t->children[ 0 ].task->parents.size() );\n";
    os << "    for( Task *p : t->children[ 0 ].task->parents ) P( p->kernel.name );\n";
    os << "    if ( ! t->move_arg( 0 ) )\n";
    os << "        ERROR( \"not owned data\" );\n";
    os << "    return nullptr;\n";
    os << "}\n";
}
