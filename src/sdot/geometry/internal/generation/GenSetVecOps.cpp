#include "../../../support/split.h"
#include "../../../support/P.h"
#include "GenSetVecOps.h"

GenSetVecOps::GenSetVecOps( std::string func_name, TI dim ) : func_name( func_name ), dim( dim ) {
    // parse func_name
    std::vector<std::string> spl = split( func_name, "_l" );
    spl.erase( spl.begin() );
    for( std::string c : spl ) {
        std::vector<std::string> nbrs = split( c, "_" );
        TI num_ouput_shape = std::stol( nbrs[ 0 ] );
        if ( nl.size() <= num_ouput_shape )
            nl.resize( num_ouput_shape + 1 );

        Output output;
        for( TI i = 1; i < nbrs.size(); i += 2 )
            output.nodes.push_back( { TI( std::stol( nbrs[ i ] ) ), TI( std::stol( nbrs[ i + 1 ] ) ) } );

        nl[ num_ouput_shape ].outputs.push_back( output );
    }

    // nb_src_nodes
    nb_src_nodes = 0;
    for( const ByOutputShape &bos : nl )
        for( const Output &output : bos.outputs )
            for( const Node &node: output.nodes )
                for( TI ind : node )
                    nb_src_nodes = std::max( nb_src_nodes, ind + 1 );

    // needed_disjoint_pairs
    for( const ByOutputShape &bos : nl )
        for( const Output &output : bos.outputs )
            for( const Node &node: output.nodes )
                if ( node[ 0 ] != node[ 1 ] )
                    needed_disjoint_pairs.insert( { node[ 0 ], node[ 1 ] } );

    // needed_sp_inds
    for( const Node &node : needed_disjoint_pairs )
        for( TI ind : node )
            needed_sp_inds.insert( ind );
}

void GenSetVecOps::write( std::ostream &os ) const {
    os << "    template<class ShapeCoords>\n";
    os << "    static void " << func_name << "( const TI *indices_data, TI indices_size";
    for( TI num_bos = 0; num_bos < nl.size(); ++num_bos )
        os << ", ShapeCoords &nc_" << num_bos << ", std::array<TI," << nl[ num_bos ].outputs[ 0 ].nodes.size() << "> dst_" << num_bos << "_indices";
    os << ", const ShapeCoords &sc, std::array<TI," << nb_src_nodes << "> src_indices ) {\n";
    os << "        Pos pos;\n";
    os << "        Id id;\n";

    // src_pos_..._ptr
    os << "\n";
    for( TI ind = 0; ind < nb_src_nodes; ++ind )
        for( TI d = 0; d < dim; ++d )
            os << "        const TF *src_pos_" << ind << "_" << ind << "_" << d << "_ptr = sc[ pos ][ src_indices[ " << ind << " ] ][ " << d << " ].data;\n";

    // src_id_ptr
    os << "\n";
    os << "        const TI *src_id_ptr = sc[ id ].data;\n";

    //
    if ( ! needed_sp_inds.empty() ) {
        os << "\n";
        for( TI ind : needed_sp_inds )
            os << "        const TF *src_sp_" << ind << "_ptr = sc[ pos ][ src_indices[ " << ind << " ] ][ " << dim << " ].data;\n";
    }

    // dst_..._pos_..._ptr
    os << "\n";
    for( TI num_dst = 0; num_dst < nl.size(); ++num_dst )
        for( TI num_node = 0; num_node < nl[ num_dst ].outputs[ 0 ].nodes.size(); ++num_node )
            for( TI d = 0; d < dim; ++d )
                os << "        TF *dst_" << num_dst << "_pos_" << num_node << "_" << d << "_ptr = nc_" << num_dst << "[ pos ][ dst_" << num_dst << "_indices[ " << num_node << " ] ][ " << d << " ].data + nc_" << num_dst << ".size;\n";

    // dst_..._id_ptr
    os << "\n";
    for( TI num_dst = 0; num_dst < nl.size(); ++num_dst )
        os << "        TI *dst_" << num_dst << "_id_ptr = nc_" << num_dst << "[ id ].data + nc_" << num_dst << ".size;\n";

    os << "\n";
    os << "        SimdRange<SimdSize<TF,Arch>::value>::for_each( indices_size, [&]( TI beg_num_ind, auto simd_size ) {\n";
    os << "            using VI = SimdVec<TI,simd_size.value,Arch>;\n";
    os << "            using VF = SimdVec<TF,simd_size.value,Arch>;\n";
    os << "\n";
    os << "            VI inds = VI::load_aligned( indices_data + beg_num_ind );\n";

    // load sp
    if ( ! needed_sp_inds.empty() ) {
        os << "\n";
        for( TI ind : needed_sp_inds )
            os << "            VF src_sp_" << ind << " = VF::gather( src_sp_" << ind << "_ptr, inds );\n";
    }

    // load pos
    os << "\n";
    for( TI ind = 0; ind < nb_src_nodes; ++ind )
        for( TI d = 0; d < dim; ++d )
            os << "            VF src_pos_" << ind << "_" << ind << "_" << d << " = VF::gather( src_pos_" << ind << "_" << ind << "_" << d << "_ptr, inds );\n";

    // mult coeffs
    if ( ! needed_disjoint_pairs.empty() ) {
        os << "\n";
        for( const Node &node : needed_disjoint_pairs )
            os << "            VF m_" << node[ 0 ] << "_" << node[ 1 ] << " = src_sp_" << node[ 0 ] << " / ( src_sp_" << node[ 0 ] << " - src_sp_" << node[ 1 ] << " );\n";

        os << "\n";
        for( const Node &node : needed_disjoint_pairs )
            for( TI d = 0; d < dim; ++d )
                os << "            VF src_pos_" << node[ 0 ] << "_" << node[ 1 ] << "_" << d << " = src_pos_" << node[ 0 ] << "_" << node[ 0 ] << "_" << d
                   << " + m_" << node[ 0 ] << "_" << node[ 1 ] << " * ( src_pos_" << node[ 1 ] << "_" << node[ 1 ] << "_" << d << " - src_pos_" << node[ 0 ] << "_" << node[ 0 ] << "_" << d << " );\n";
    }

    // load id
    os << "\n";
    os << "            VI ids = VI::gather( src_id_ptr, inds );\n";

    // compute + store
    os << "\n";
    for( TI num_dst_list = 0; num_dst_list < nl.size(); ++num_dst_list )
        for( TI num_in_dst_list = 0; num_in_dst_list < nl[ num_dst_list ].outputs.size(); ++num_in_dst_list )
            for( TI num_node = 0; num_node < nl[ num_dst_list ].outputs[ num_in_dst_list ].nodes.size(); ++num_node )
                for( TI d = 0; d < dim; ++d )
                    os << "            VF::store( dst_" << num_dst_list << "_pos_" << num_node << "_" << d << "_ptr + "
                       << nl[ num_dst_list ].outputs.size() << " * beg_num_ind + " << num_in_dst_list << " * simd_size.value, src_pos_"
                       << nl[ num_dst_list ].outputs[ num_in_dst_list ].nodes[ num_node ][ 0 ] << "_"
                       << nl[ num_dst_list ].outputs[ num_in_dst_list ].nodes[ num_node ][ 1 ] << "_" << d << " );\n";

    os << "\n";
    for( TI num_dst_list = 0; num_dst_list < nl.size(); ++num_dst_list )
        for( TI num_in_dst_list = 0; num_in_dst_list < nl[ num_dst_list ].outputs.size(); ++num_in_dst_list )
            os << "            VI::store( dst_" << num_dst_list << "_id_ptr + " << nl[ num_dst_list ].outputs.size() << " * beg_num_ind + " << num_in_dst_list << " * simd_size.value, ids );\n";

    os << "        } );\n";
    os << "\n";
    for( TI num_dst_list = 0; num_dst_list < nl.size(); ++num_dst_list )
        os << "        nc_" << num_dst_list << ".size += indices_size * " << nl[ num_dst_list ].outputs.size() << ";\n";
    os << "    }\n";
}
