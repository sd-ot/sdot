#include "GlobGeneGeomData.h"
#include <fstream>

namespace sdot {

std::string GlobGeneGeomData::mk_item_name( std::vector<TI> inds ) {
    needed_cut_ops.insert( inds );

    std::string res = "mk_items";
    for( TI i : inds )
        res += "_" + std::to_string( i );
    return res;
}

void GlobGeneGeomData::write_gen_decl( std::ostream &os, std::vector<TI> inds, std::string prefix, std::string suffix ) {
    os << prefix << "void " << mk_item_name( inds ) << "( ";
    os << "ShapeData &new_shape_data, const std::array<BI," << 3 << "> &new_node_indices, ";
    os << "const ShapeData &old_shape_data, const std::array<BI,3> &old_node_indices, BI num_case, const void *cut_ids, N<2> dim )" << suffix;
}

void GlobGeneGeomData::write_gen_decls( std::string filename ) {
    std::ofstream os( filename );
    for( std::vector<TI> inds : needed_cut_ops )
        write_gen_decl( os, inds, "virtual ", " = 0;\n" );
}

void GlobGeneGeomData::write_gen_defs( std::string filename, bool /*gpu*/ ) {
    std::ofstream os( filename );

    for( std::vector<TI> inds : needed_cut_ops ) {
        write_gen_decl( os, inds, {}, " override {\n" );

        os << "    TF *new_x_0 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 0 ] * dim + 0 ) * new_shape_data.rese;\n";
        os << "    TF *new_y_0 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 0 ] * dim + 1 ) * new_shape_data.rese;\n";
        os << "    TF *new_x_1 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 1 ] * dim + 0 ) * new_shape_data.rese;\n";
        os << "    TF *new_y_1 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 1 ] * dim + 1 ) * new_shape_data.rese;\n";
        os << "    TF *new_x_2 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 2 ] * dim + 0 ) * new_shape_data.rese;\n";
        os << "    TF *new_y_2 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 2 ] * dim + 1 ) * new_shape_data.rese;\n";
        os << "\n";
        os << "    TI *new_f_0 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 0 ] * new_shape_data.rese;\n";
        os << "    TI *new_f_1 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 1 ] * new_shape_data.rese;\n";
        os << "    TI *new_f_2 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 2 ] * new_shape_data.rese;\n";
        os << "\n";
        os << "    TI *new_ids = reinterpret_cast<TI *>( new_shape_data.ids );\n";
        os << "\n";
        os << "    const TF *old_x_0 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 0 ] * dim + 0 ) * old_shape_data.rese;\n";
        os << "    const TF *old_y_0 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 0 ] * dim + 1 ) * old_shape_data.rese;\n";
        os << "    const TF *old_x_1 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 1 ] * dim + 0 ) * old_shape_data.rese;\n";
        os << "    const TF *old_y_1 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 1 ] * dim + 1 ) * old_shape_data.rese;\n";
        os << "    const TF *old_x_2 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 2 ] * dim + 0 ) * old_shape_data.rese;\n";
        os << "    const TF *old_y_2 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 2 ] * dim + 1 ) * old_shape_data.rese;\n";
        os << "\n";
        os << "    const TI *old_f_0 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 0 ] * old_shape_data.rese;\n";
        os << "    const TI *old_f_1 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 1 ] * old_shape_data.rese;\n";
        os << "    const TI *old_f_2 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 2 ] * old_shape_data.rese;\n";
        os << "\n";
        os << "    const TI *old_ids = reinterpret_cast<const TI *>( old_shape_data.ids );\n";
        os << "\n";
        os << "    const TI *o0 = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::offset_0 ] );\n";
        os << "    const TI *o1 = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::offset_1 ] );\n";
        os << "    const TI *cc = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::cut_case ] );\n";
        os << "\n";
        os << "    for( BI nmp = 0; nmp < nb_multiprocs(); ++nmp ) {\n";
        os << "        for( BI ind = o0[ num_case * nb_multiprocs() + nmp ]; ind < o1[ num_case * nb_multiprocs() + nmp ]; ++ind ) {\n";
        os << "            TI off = cc[ ind ];\n";
        os << "\n";
        os << "            new_x_0[ new_shape_data.size ] = old_x_0[ off ];\n";
        os << "            new_y_0[ new_shape_data.size ] = old_y_0[ off ];\n";
        os << "            new_x_1[ new_shape_data.size ] = old_x_1[ off ];\n";
        os << "            new_y_1[ new_shape_data.size ] = old_y_1[ off ];\n";
        os << "            new_x_2[ new_shape_data.size ] = old_x_2[ off ];\n";
        os << "            new_y_2[ new_shape_data.size ] = old_y_2[ off ];\n";
        os << "\n";
        os << "            new_f_0[ new_shape_data.size ] = old_f_0[ off ];\n";
        os << "            new_f_1[ new_shape_data.size ] = old_f_1[ off ];\n";
        os << "            new_f_2[ new_shape_data.size ] = old_f_2[ off ];\n";
        os << "\n";
        os << "            new_ids[ new_shape_data.size ] = old_ids[ off ];\n";
        os << "\n";
        os << "            ++new_shape_data.size;\n";
        os << "        }\n";
        os << "    }\n";
        os << "}\n";
    }
}

} // namespace sdot

