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

    // nb_src_nodes, needed_src_inds
    nb_src_nodes = 0;
    for( const ByOutputShape &bos : nl ) {
        for( const Output &output : bos.outputs ) {
            for( const Node &node: output.nodes ) {
                for( TI ind : node ) {
                    nb_src_nodes = std::max( nb_src_nodes, ind + 1 );
                    needed_src_inds.insert( ind );
                }
            }
        }
    }
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
    for( TI ind : needed_src_inds )
        for( TI d = 0; d < dim; ++d )
            os << "        const TF *src_pos_" << ind << "_" << ind << "_" << d << "_ptr = sc[ pos ][ src_indices[ " << ind << " ] ][ " << d << " ].data;\n";

    // src_id_ptr
    os << "\n";
    os << "        const TI *src_id_ptr = sc[ id ].data;\n";

    //
    os << "\n";
    for( TI ind : needed_src_inds )
        os << "        const TF *src_sp_" << ind << "_ptr = sc[ pos ][ src_indices[ " << ind << " ] ][ " << dim << " ].data;\n";

    // dst_..._pos_..._ptr
    os << "\n";
    for( TI num_dst = 0; num_dst < nl.size(); ++num_dst )
        for( TI num_node = 0; num_node < nl[ num_dst ].outputs[ 0 ].nodes.size(); ++num_node )
            for( TI d = 0; d < dim; ++d )
                os << "        TF *dst_" << num_dst << "_pos_" << num_node << "_" << dim << "_ptr = nc_" << num_dst << "[ pos ][ dst_" << num_dst << "_indices[ " << num_node << " ] ][ " << dim << " ].data + nc_" << num_dst << ".size;\n";

    // dst_..._id_ptr
    os << "\n";
    for( TI num_dst = 0; num_dst < nl.size(); ++num_dst )
        os << "        TI *dst_" << num_dst << "_id_ptr = nc_" << num_dst << "[ id ].data + nc_" << num_dst << ".size;\n";


    os << "    }\n";
}
