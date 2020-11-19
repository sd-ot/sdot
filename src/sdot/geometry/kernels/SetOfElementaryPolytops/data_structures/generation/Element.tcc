#include "Element.h"
#include "CutCase.h"

inline Element::Element( std::string name ) {
    nvi = 2;

    nb_nodes = 0;
    std::size_t p = 0;
    for( ; p < name.size() && std::isdigit( name[ p ] ); ++p )
        nb_nodes += nb_nodes * 10 + ( name[ p ] - '0' );

    nb_faces = nb_nodes;
}

inline void Element::write_vtk_info( std::ostream &os, std::string var_name) {
    auto disp_single_elem = [&]( const char *name ) {
        os << "    " << var_name << ".vtk_elements = { { " << name << ", { ";
        for( int i = 0; i < nb_nodes; ++i )
            os << ( i ? ", " : "" ) << i;
        os << " } } };\n";
    };

    // 2D
    switch ( nb_nodes ) {
    case 3 : disp_single_elem( "5" ); break;
    case 4 : disp_single_elem( "9" ); break;
    default: disp_single_elem( "7" ); break;
    }
}

inline void Element::write_cut_info( std::ostream &os, std::string var_name, std::map<std::string,Element> &elements ) {
    // init of cut_cases
    std::vector<CutCase> cut_cases( std::uint64_t( 1 ) << nb_nodes );
    for( std::size_t n = 0; n < cut_cases.size(); ++n ) {
        std::vector<bool> outside( nb_nodes );
        for( int j = 0; j < nb_nodes; ++j )
            outside[ j ] = n & ( std::uint64_t( 1 ) << j );
        cut_cases[ n ].init( *this, outside, elements );
    }

    // tot_sub_cases
    std::size_t tot_sub_cases = 0;
    for( CutCase &cut_case : cut_cases )
        tot_sub_cases += cut_case.possibilities.size();

    // write nb_sub_cases (for each case)
    os << "    " << var_name << ".cut_info.nb_sub_cases = {";
    for( std::size_t n = 0; n < cut_cases.size(); ++n )
        os << ( n ? ", " : " " ) << cut_cases[ n ].possibilities.size();
    os << " };\n";

    // write nb_output_elements
    std::map<std::string,std::vector<std::size_t>> nb_output_elements;
    for( std::size_t i = 0, sc = 0; i < cut_cases.size(); ++i ) {
        for( std::size_t j = 0; j < cut_cases[ i ].possibilities.size(); ++j, ++sc ) {
            for( CutItem &cut_item : cut_cases[ i ].possibilities[ j ]->cut_op.cut_items ) {
                nb_output_elements[ cut_item.name() ].resize( tot_sub_cases, 0 );
                nb_output_elements[ cut_item.name() ][ sc ]++;
            }
        }
    }
    os << "    " << var_name << ".cut_info.nb_output_elements = {";
    std::size_t cpt = 0;
    for( const auto &p : nb_output_elements ) {
        os << ( cpt++ ? "," : "" ) << "\n        { \"" << p.first << "\", {";
        for( std::size_t n = 0; n < p.second.size(); ++n )
            os << ( n ? ", " : " " ) << p.second[ n ];
        os << " } }";
    }
    os << "\n    };\n";

    // write new_elems [std::string operation_name; std::vector<OutCutOp> outputs; std::vector<TI> inp_node_corr, inp_face_corr; TI num_case, num_sub_case]
    os << "    " << var_name << ".cut_info.new_elems = {";
    for( std::size_t num_cut_case = 0, cs = 0; num_cut_case < cut_cases.size(); ++num_cut_case ) {
        CutCase &cut_case = cut_cases[ num_cut_case ];
        for( std::size_t num_poss = 0; num_poss < cut_case.possibilities.size(); ++num_poss ) {
            std::unique_ptr<CutOpWithNamesAndInds> &cownai = cut_case.possibilities[ num_poss ];
            os << ( cs++ ? "," : "" ) << "\n        { " << "\"" << /* operation_name */ cownai->cut_op.mk_item_func_name() << "\", { "; /*std::vector<OutCutOp> outputs*/
            for( std::size_t num_output = 0; num_output < cownai->outputs.size(); ++num_output ) {
                CutOpWithNamesAndInds::Out &output = cownai->outputs[ num_output ];
                os << ( num_output ? ", " : "" ) << "{ \"" << /*shape_type*/ output.shape_name << "\", {" /* node_corr */; //  OutCutOp { std::string shape_type; std::vector<TI> node_corr, face_corr; };
                for( std::size_t n = 0; n < output.output_node_inds.size(); ++n )
                    os << ( n ? ", " : " " ) << output.output_node_inds[ n ];
                os << " }, {"; /* face_corr */
                for( std::size_t n = 0; n < output.output_face_inds.size(); ++n )
                    os << ( n ? ", " : " " ) << output.output_face_inds[ n ];
                os << " } }";
            }
            os << " }, {"; /* inp_node_corr */
            for( std::size_t n = 0; n < cownai->input_node_inds.size(); ++n )
                os << ( n ? ", " : " " ) << cownai->input_node_inds[ n ];
            os << " }, {"; /* inp_face_corr */
            for( std::size_t n = 0; n < cownai->input_face_inds.size(); ++n )
                os << ( n ? ", " : " " ) << cownai->input_face_inds[ n ];
            os << " }, " << /* num_case */ num_cut_case << ", " << /* num_sub_case */ num_poss;

            os << " }";
        }
    }

    os << "\n    };\n";
}
