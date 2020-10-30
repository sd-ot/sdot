#include "GlobGeneGeomData.h"
#include <fstream>
#include <array>

namespace sdot {

void GlobGeneGeomData::write_gen_decl( std::ostream &os, const CutOp &cut_op, std::string prefix, std::string suffix ) {
    os << prefix << "void " << cut_op.mk_item_func_name() << "( ";
    for( TI i = 0; i < cut_op.cut_items.size(); ++i )
        os << "ShapeData &nsd_" << i << ", const std::array<BI," << cut_op.cut_items[ i ].nodes.size() << "> &nni_" << i << ", const std::array<BI," << cut_op.cut_items[ i ].faces.size() << "> &nfi_" << i << ", ";
    os << "const ShapeData &osd, const std::array<BI," << cut_op.nb_input_nodes() << "> &oni, const std::array<BI," << cut_op.nb_input_faces() << "> &ofi, BI num_case, const void *cut_ids, N<" << cut_op.dim << "> dim )" << suffix;
}

void GlobGeneGeomData::write_gen_decls( std::string filename ) {
    std::ofstream os( filename );
    for( const CutOp &cut_op : needed_cut_ops ) {
        if ( cut_op.cut_items.empty() )
            continue;
        write_gen_decl( os, cut_op, "virtual ", " = 0;\n" );
    }
}

void GlobGeneGeomData::write_gen_defs( std::string filename, bool /*gpu*/ ) {
    static const char *nd = "xyzabcdefghijklmnopqrstuv";
    std::ofstream os( filename );

    for( const CutOp &cut_op : needed_cut_ops ) {
        if ( cut_op.cut_items.empty() )
            continue;
        write_gen_decl( os, cut_op, {}, " override {\n" );

        // ptr to new items
        for( TI no = 0; no < cut_op.cut_items.size(); ++no )
            for( TI nn = 0; nn < cut_op.cut_items[ no ].nodes.size(); ++nn )
                for( TI d = 0; d < cut_op.dim; ++d )
                    os << "    TF *new_" << nd[ d ] << "_" << nn << "_" << no << " = reinterpret_cast<TF *>( nsd_" << no << ".coordinates ) + ( nni_" << no << "[ " << nn << " ] * dim + " << d << " ) * nsd_" << no << ".rese;\n";
        os << "\n";

        for( TI no = 0; no < cut_op.cut_items.size(); ++no )
            for( TI nn = 0; nn < cut_op.cut_items[ no ].nodes.size(); ++nn )
                os << "    TI *new_f_" << nn << "_" << no << " = reinterpret_cast<TI *>( nsd_" << no << ".face_ids ) + nni_" << no << "[ " << nn << " ] * nsd_" << no << ".rese;\n";
        os << "\n";

        for( TI no = 0; no < cut_op.cut_items.size(); ++no )
            os << "    TI *new_ids_" << no << " = reinterpret_cast<TI *>( nsd_" << no << ".ids );\n";
        os << "\n";

        // ptr to old items
        for( TI nn = 0; nn < cut_op.nb_input_nodes(); ++nn )
            for( TI d = 0; d < cut_op.dim; ++d )
                os << "    const TF *old_" << nd[ d ] << "_" << nn << " = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ " << nn << " ] * dim + " << d << " ) * osd.rese;\n";
        os << "\n";

        for( TI nn = 0; nn < cut_op.nb_input_faces(); ++nn )
            os << "    const TI *old_f_" << nn << " = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ " << nn << " ] * osd.rese;\n";
        os << "\n";

        os << "    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );\n";
        os << "\n";

        // ptr to indices
        os << "    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );\n";

        // needed intersection points
        std::set<std::array<TI,2>> cs;
        for( TI no = 0; no < cut_op.cut_items.size(); ++no )
            for( auto nn : cut_op.cut_items[ no ].nodes )
                if ( nn[ 0 ] != nn[ 1 ] )
                    cs.insert( { nn[ 0 ], nn[ 1 ] } );

        if ( cs.size() ) {
            os << "\n";
            for( TI nn = 0; nn < cut_op.nb_input_nodes(); ++nn )
                os << "    const TF *old_scp_" << nn << " = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ " << nn << " ] * osd.rese;\n";
        }

        // loop over indices
        os << "\n";
        os << "    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {\n";
        os << "        TI off = indices[ num_ind ];\n";

        // needed scps
        std::set<TI> needed_scps;
        for( std::array<TI,2> c : cs )
            for( TI v : c )
                needed_scps.insert( v );

        if ( needed_scps.size() ) {
            os << "\n";
            for( TI nn : needed_scps )
                os << "    TF scp_" << nn << " = old_scp_" << nn << "[ off ];\n";
        }

        // compute the new node positions
        os << "\n";
        for( TI nn = 0; nn < cut_op.nb_input_nodes(); ++nn )
            for( TI d = 0; d < cut_op.dim; ++d )
                os << "        TF " << nd[ d ] << "_" << nn << "_" << nn << " = old_" << nd[ d ] << "_" << nn << "[ off ];\n";
        for( std::array<TI,2> c : cs )
            for( TI d = 0; d < cut_op.dim; ++d )
                os << "        TF " << nd[ d ] << "_" << c[ 0 ] << "_" << c[ 1 ] << " = " << nd[ d ] << "_" << c[ 0 ] << "_" << c[ 0 ] << " + scp_" << c[ 0 ] << " / ( scp_" << c[ 0 ] << " - scp_" << c[ 1 ] << " ) * ( " << nd[ d ] << "_" << c[ 1 ] << "_" << c[ 1 ] << " - " << nd[ d ] << "_" << c[ 0 ] << "_" << c[ 0 ] << " );\n";
        os << "\n";

        // store them
        for( TI no = 0; no < cut_op.cut_items.size(); ++no )
            for( TI nn = 0; nn < cut_op.cut_items[ no ].nodes.size(); ++nn )
                for( TI d = 0; d < cut_op.dim; ++d )
                    os << "        new_" << nd[ d ] << "_" << nn << "_" << no << "[ nsd_" << no << ".size ] = " << nd[ d ] << "_" << cut_op.cut_items[ no ].nodes[ nn ][ 0 ] << "_" << cut_op.cut_items[ no ].nodes[ nn ][ 1 ] << ";\n";
        os << "\n";

        for( TI no = 0; no < cut_op.cut_items.size(); ++no )
            for( TI nn = 0; nn < cut_op.cut_items[ no ].faces.size(); ++nn )
                if ( cut_op.cut_items[ no ].faces[ nn ] == TI( -1 ) )
                    os << "        new_f_" << nn << "_" << no << "[ nsd_" << no << ".size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];\n";
                else
                    os << "        new_f_" << nn << "_" << no << "[ nsd_" << no << ".size ] = old_f_" << cut_op.cut_items[ no ].faces[ nn ] << "[ off ];\n";
        os << "\n";

        for( TI no = 0; no < cut_op.cut_items.size(); ++no )
            os << "        new_ids_" << no << "[ nsd_" << no << ".size ] = old_ids[ off ];\n";
        os << "\n";

        for( TI no = 0; no < cut_op.cut_items.size(); ++no )
            os << "        ++nsd_" << no << ".size;\n";
        os << "    }\n";
        os << "}\n";
    }
}

} // namespace sdot

