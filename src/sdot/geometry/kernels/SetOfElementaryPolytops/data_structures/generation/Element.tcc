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

    // write nb_sub_cases (for each case)
    os << "    " << var_name << ".cut_info.nb_sub_cases = {";
    for( std::size_t n = 0; n < cut_cases.size(); ++n )
        os << ( n ? ", " : " " ) << cut_cases[ n ].possibilities.size();
    os << " };\n";

    // write nb_output_elements
    std::map<std::string,std::vector<std::vector<std::size_t>>> nb_output_elements;
    for( const auto &p : elements ) {
        std::vector<std::vector<std::size_t>> &vo = nb_output_elements[ p.first ];
        vo.resize( cut_cases.size() );
        for( std::size_t num_case = 0; num_case < cut_cases.size(); ++num_case )
            vo[ num_case ].resize( cut_cases[ num_case ].possibilities.size(), 0 );
    }
    for( std::size_t num_case = 0; num_case < cut_cases.size(); ++num_case )
        for( std::size_t num_sub_case = 0; num_sub_case < cut_cases[ num_case ].possibilities.size(); ++num_sub_case )
            for( CutItem &cut_item : cut_cases[ num_case ].possibilities[ num_sub_case ]->cut_op.cut_items )
                nb_output_elements[ cut_item.name() ][ num_case ][ num_sub_case ]++;

    os << "    " << var_name << ".cut_info.nb_output_elements = {";
    std::size_t cpt = 0;
    for( const auto &p : nb_output_elements ) {
        os << ( cpt++ ? "," : "" ) << "\n        { \"" << p.first << "\", {";
        for( std::size_t num_case = 0; num_case < p.second.size(); ++num_case ) {
            os << ( num_case ? ", " : " " ) << "{";
            for( std::size_t num_sub_case = 0; num_sub_case < p.second[ num_case ].size(); ++num_sub_case )
                os << ( num_sub_case ? ", " : " " ) << p.second[ num_case ][ num_sub_case ];
            os << " }";
        }
        os << " } }";
    }
    os << "\n    };\n";

    // write new_elems [std::string operation_name; std::vector<OutCutOp> outputs; std::vector<TI> inp_node_corr, inp_face_corr; TI num_case, num_sub_case]
    os << "    " << var_name << ".cut_info.new_elems = {";
    for( std::size_t num_case = 0; num_case < cut_cases.size(); ++num_case ) {
        CutCase &cut_case = cut_cases[ num_case ];

        os << ( num_case  ? "," : "" ) << "\n        {";
        for( std::size_t num_sub_case = 0; num_sub_case < cut_case.possibilities.size(); ++num_sub_case ) {
            std::unique_ptr<CutOpWithNamesAndInds> &cownai = cut_case.possibilities[ num_sub_case ];
            os << ( num_sub_case  ? "," : "" ) << " /*sub case " << num_sub_case << ":*/ { " << "\"" << /* operation_name */ cownai->cut_op.mk_item_func_name() << "\", { "; /*std::vector<OutCutOp> outputs*/
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
            os << " }";

            os << " }";
        }
        os << " }";
    }
    os << "\n    };\n";

    // write lengths
    os << "    " << var_name << ".cut_info.lengths = {";
    for( std::size_t num_case = 0; num_case < cut_cases.size(); ++num_case ) {
        CutCase &cut_case = cut_cases[ num_case ];
        os << ( num_case  ? "," : "" ) << "\n        {";
        for( std::size_t num_sub_case = 0; num_sub_case < cut_case.possibilities.size(); ++num_sub_case ) {
            os << ( num_sub_case  ? "," : "" ) << " /*sub case " << num_sub_case << ":*/ { ";
            std::size_t num_output = 0;
            cut_case.possibilities[ num_sub_case ]->for_each_new_edge( [&]( std::size_t n00, std::size_t n01, std::size_t n10, std::size_t n11 ) {
                os << ( num_output++ ? ", " : "" ) << "{ " << n00 << ", " << n01 << ", " << n10 << ", " << n11 << " }";
            } );
            os << " }";
        }
        os << " }";
    }
    os << "\n    };\n";
}
