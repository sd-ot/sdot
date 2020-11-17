#include <sstream>
#include <vector>
#include <string>
#include <array>
#include <set>
using TI = std::size_t;

struct ParmReader {
    struct Output {
        std::vector<std::array<int,2>> inp_nodes;
        std::vector<int> inp_faces;
    };

    ParmReader( std::string parm ) {
        // dim nb_nodes_0 0 0 1 1 2 2 nb_faces_0 0 1 2 nb_nodes_1 ...
        std::istringstream ss( parm );
        ss >> dim;

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

    std::set<int> needed_input_nodes() const {
        std::set<int> res;
        for( const Output &output : outputs )
            for( std::array<int,2> a : output.inp_nodes )
                for( int n : a )
                    res.insert( n );
        return res;
    }

    std::set<int> needed_input_faces() const {
        std::set<int> res;
        for( const Output &output : outputs )
                for( int n : output.inp_faces )
                    if ( n >= 0 )
                        res.insert( n );
        return res;
    }

    std::vector<Output> outputs;
    int dim;
};

void mk_items( std::ostream &os, const std::string &kernel_name, const std::string &parameter ) {
    static const char *nd = "xyzabcdefghijklmnopqrstuv";
    ParmReader pr( parameter );

    os << "#include <sdot/geometry/ShapeCutTmpData.h>\n";
    os << "#include <sdot/geometry/ShapeMap.h>\n";
    os << "#include <parex/support/P.h>\n";
    os << "#include <parex/TaskRef.h>\n";
    os << "using namespace parex;\n";
    os << "using namespace sdot;\n";
    os << "\n";
    os << "// " << parameter << "\n";

    os << "template<class TF,class TI,int dim,class VI,class VO>\n";
    os << "ShapeMap<TF,TI,dim> *" << kernel_name << "( Task *t, ShapeMap<TF,TI,dim> &out_shape_map";
    for( TI no = 0; no < pr.outputs.size(); ++no )
        os << ", ShapeType &out_shape_type_" << no
           << ", const VI &out_node_corr_" << no
           << ", const VI &out_face_corr_" << no << "\n";
    os << ", const VI &inp_node_corr, const VI &inp_face_corr, TI beg_ind, TI end_ind, std::map<ShapeType *,ShapeCutTmpData<TF,TI>> &tmp_cut_data_map, ShapeMap<TF,TI,dim> &inp_shape_map, ShapeType &inp_shape_type, const VO &new_face_ids ) {\n";
    os << "    P( t->children[ 0 ].task->kernel.name );\n";
    os << "    P( t->children[ 0 ].task->ref_count );\n";
    os << "    if ( ! t->move_arg( 0 ) )\n";
    os << "        ERROR( \"not owned data\" );\n";

    os << "\n";
    for( TI no = 0; no < pr.outputs.size(); ++no )
        os << "    ShapeData<TF,TI,dim> &out_shape_data_" << no << " = out_shape_map.map.find( &out_shape_type_" << no << " )->second;\n";
    os << "    ShapeData<TF,TI,dim> &inp_shape_data = inp_shape_map.map.find( &inp_shape_type )->second;\n";

    // ptr to new items
    os << "\n";
    for( TI no = 0; no < pr.outputs.size(); ++no )
        for( TI nn = 0; nn < pr.outputs[ no ].inp_nodes.size(); ++nn )
            for( int d = 0; d < pr.dim; ++d )
                os << "    TF *new_" << nd[ d ] << "_" << nn << "_" << no << " = out_shape_data_" << no << ".coordinates.ptr( out_node_corr_" << no << "[ " << nn << " ] * dim + " << d << " );\n";

    os << "\n";
    for( TI no = 0; no < pr.outputs.size(); ++no )
        for( TI nn = 0; nn < pr.outputs[ no ].inp_faces.size(); ++nn )
            os << "    TI *new_f_" << nn << "_" << no << " = out_shape_data_" << no << ".face_ids.ptr( out_face_corr_" << no << "[ " << nn << " ] );\n";

    os << "\n";
    for( TI no = 0; no < pr.outputs.size(); ++no )
        os << "    TI *new_ids_" << no << " = out_shape_data_" << no << ".ids.ptr();\n";

    // ptr to old items
    os << "\n";
    for( TI nn : pr.needed_input_nodes() )
        for( int d = 0; d < pr.dim; ++d )
            os << "    const TF *old_" << nd[ d ] << "_" << nn << " = inp_shape_data.coordinates.ptr( inp_node_corr[ " << nn << " ] * dim + " << d << " ) ;\n";

    os << "\n";
    for( TI nn : pr.needed_input_faces() )
        os << "    const TI *old_f_" << nn << " = inp_shape_data.face_ids.ptr( inp_face_corr[ " << nn << " ] );\n";

    os << "\n";
    os << "    const TI *old_ids = inp_shape_data.ids.ptr();\n";

    // ptr to indices
    os << "\n";
    os << "    ShapeCutTmpData<TF,TI> &tmp_cut_data = tmp_cut_data_map.find( &inp_shape_type )->second;\n";
    os << "    const TI *indices = tmp_cut_data.indices.ptr();\n";

    // needed intersection points
    std::set<std::array<int,2>> cs;
    std::set<int> edge_points;
    for( TI no = 0; no < pr.outputs.size(); ++no ) {
        for( std::array<int,2> nn : pr.outputs[ no ].inp_nodes ) {
            if ( nn[ 0 ] != nn[ 1 ] ) {
                cs.insert( { nn[ 0 ], nn[ 1 ] } );
                for( int n : nn )
                    edge_points.insert( n );
            }
        }
    }

    if ( edge_points.size() ) {
        os << "\n";
        for( int n : edge_points )
            os << "    const TF *inp_scp_" << n << " = tmp_cut_data.scalar_products.ptr( inp_node_corr[ " << n << " ] );\n";
    }

    // loop over indices
    os << "\n";
    os << "    for( TI num_ind = beg_ind; num_ind < end_ind; ++num_ind ) {\n";
    os << "        TI index = indices[ num_ind ];\n";

    // edge_points
    if ( edge_points.size() ) {
        os << "\n";
        for( int nn : edge_points )
            os << "        TF scp_" << nn << " = inp_scp_" << nn << "[ index ];\n";

        os << "\n";
        for( std::array<int,2> c : cs )
            os << "        TF d_" << c[ 0 ] << "_" << c[ 1 ] << " = scp_" << c[ 0 ] << " / ( scp_" << c[ 0 ] << " - scp_" << c[ 1 ] << " );\n";
    }

    // compute the new node positions
    os << "\n";
    for( int nn : pr.needed_input_nodes() )
        for( int d = 0; d < pr.dim; ++d )
            os << "        TF " << nd[ d ] << "_" << nn << "_" << nn << " = old_" << nd[ d ] << "_" << nn << "[ index ];\n";

    os << "\n";
    for( std::array<int,2> c : cs )
        for( int d = 0; d < pr.dim; ++d )
            os << "        TF " << nd[ d ] << "_" << c[ 0 ] << "_" << c[ 1 ] << " = " << nd[ d ] << "_" << c[ 0 ] << "_" << c[ 0 ] << " + d_" << c[ 0 ] << "_" << c[ 1 ] << " * ( " << nd[ d ] << "_" << c[ 1 ] << "_" << c[ 1 ] << " - " << nd[ d ] << "_" << c[ 0 ] << "_" << c[ 0 ] << " );\n";

    // new indices
    os << "\n";
    for( TI no = 0; no < pr.outputs.size(); ++no )
        os << "        TI ni_" << no << " = out_shape_data_" << no << ".size++;\n";

    // store the points
    os << "\n";
    for( TI no = 0; no < pr.outputs.size(); ++no )
        for( TI nn = 0; nn < pr.outputs[ no ].inp_nodes.size(); ++nn )
            for( int d = 0; d < pr.dim; ++d )
                os << "        new_" << nd[ d ] << "_" << nn << "_" << no << "[ ni_" << no << " ] = " << nd[ d ] << "_" << pr.outputs[ no ].inp_nodes[ nn ][ 0 ] << "_" << pr.outputs[ no ].inp_nodes[ nn ][ 1 ] << ";\n";

    // store the faces
    os << "\n";
    for( TI no = 0; no < pr.outputs.size(); ++no ) {
        for( TI nn = 0; nn < pr.outputs[ no ].inp_faces.size(); ++nn )
            if ( pr.outputs[ no ].inp_faces[ nn ] == -1 )
                os << "        new_f_" << nn << "_" << no << "[ ni_" << no << " ] = new_face_ids[ old_ids[ index ] ];\n";
            else if ( pr.outputs[ no ].inp_faces[ nn ] == -2 )
                os << "        new_f_" << nn << "_" << no << "[ ni_" << no << " ] = TI( -1 );\n";
            else
                os << "        new_f_" << nn << "_" << no << "[ ni_" << no << " ] = old_f_" << pr.outputs[ no ].inp_faces[ nn ] << "[ index ];\n";
    }

    // store the ids
    os << "\n";
    for( TI no = 0; no < pr.outputs.size(); ++no )
        os << "        new_ids_" << no << "[ ni_" << no << " ] = old_ids[ index ];\n";
    os << "    }\n";


    os << "    return nullptr;\n";
    os << "}\n";
}
