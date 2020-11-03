#include "../support/generic_ostream_output.h"
#include "../support/TODO.h"
#include "../support/P.h"
#include "NamedRecursivePolytop.h"
#include "GlobGeneGeomData.h"
#include "CutCase.h"
using TI = std::size_t;

namespace sdot {

void NamedRecursivePolytop::write_primitive_shape_incl( std::ostream &os ) const {
    os << "// generated file\n";
    os << "#pragma once\n";
    os << "\n";
    os << "#include \"../ShapeType.h\"\n";
    os << "\n";
    os << "namespace sdot {\n";
    os << "\n";
    os << "ShapeType *s" << name.substr( 1 ) << "();\n";
    os << "\n";
    os << "}\n";
}

void NamedRecursivePolytop::write_primitive_shape_impl( std::ostream &os, GlobGeneGeomData &gggd, const std::vector<NamedRecursivePolytop> &available_primitive_shapes ) const {
    //
    std::vector<CutCase> cut_cases( std::uint64_t( 1 ) << polytop.points.size() );
    for( TI n = 0; n < cut_cases.size(); ++n ) {
        std::vector<bool> outside( polytop.points.size() );
        for( std::uint64_t j = 0; j < polytop.points.size(); ++j )
            outside[ j ] = n & ( TI( 1 ) << j );
        cut_cases[ n ].init( *this, outside, available_primitive_shapes );
    }

    //
    os << "// generated file\n";
    os << "#include \"../../kernels/VecTI.h\"\n";
    os << "#include \"../ShapeData.h\"\n";
    os << "#include \"../VtkOutput.h\"\n";
    os << "#include <iostream>\n";
    for( const NamedRecursivePolytop &ps : available_primitive_shapes )
        os << "#include \"" << ps.name << ".h\"\n";
    os << "\n";

    os << "namespace sdot {\n";
    os << "\n";

    // class decl
    os << "class " << name << " : public ShapeType {\n";
    os << "public:\n";
    os << "    virtual std::vector<BI> cut_poss_count() const override;\n";
    os << "    virtual void            display_vtk   ( VtkOutput &vo, const double **tfs, const BI **tis, unsigned dim, BI nb_items, VtkOutput::Pt *offsets ) const override;\n";
    os << "    virtual void            cut_rese      ( const std::function<void(const ShapeType *,BI)> &fc, KernelSlot *ks, const ShapeData &sd ) const override;\n";
    os << "    virtual unsigned        nb_nodes      () const override { return " << polytop.points.size() << "; }\n";
    os << "    virtual unsigned        nb_faces      () const override { return " << polytop.nb_faces() << "; }\n";
    os << "    virtual void            cut_ops       ( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const override;\n";
    os << "    virtual std::string     name          () const override { return \"" << name << "\"; }\n";
    os << "};\n";
    os << "\n";

    // definitions
    write_cut_p_c( os, cut_cases );
    write_cut_ops( os, gggd, cut_cases );
    write_dsp_vtk( os );
    write_cut_cnt( os, cut_cases );

    // singleton
    os << "\n";
    os << "\n";
    os << "// =======================================================================================\n";
    os << "ShapeType *s" << name.substr( 1 ) << "() {\n";
    os << "    static " << name << " res;\n";
    os << "    return &res;\n";
    os << "}\n";
    os << "\n";
    os << "} // namespace sdot\n";


}

void NamedRecursivePolytop::write_cut_p_c( std::ostream &os, std::vector<CutCase> &cut_cases ) const {
    os << "\n";
    os << "std::vector<ShapeType::BI> " << name << "::cut_poss_count() const {\n";
    os << "    return {";

    for( std::size_t n = 0; n < cut_cases.size(); ++n )
        os << ( n ? ", " : " " ) << cut_cases[ n ].possibilities.size();
    os << " };\n";

    os << "}\n";
}

void NamedRecursivePolytop::write_cut_ops( std::ostream &os, GlobGeneGeomData &gggd, std::vector<CutCase> &cut_cases ) const {
    os << "\n";
    os << "void " << name << "::cut_ops( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const {\n";

    // get (needed) output shape ptrs
    std::set<std::string> output_shape_names;
    for( CutCase &cut_case : cut_cases )
        for( std::unique_ptr<CutOpWithNamesAndInds> &cownai : cut_case.possibilities )
            for( CutOpWithNamesAndInds::Out &output : cownai->outputs )
                output_shape_names.insert( output.shape_name );
    for( std::string output_shape_name : output_shape_names )
        os << "    ShapeData &nsd_" << output_shape_name << " = new_shape_map.find( s" << output_shape_name.substr( 1 ) << "() )->second;\n";
    os << "\n";

    // reg needed functions in global generated code info
    for( CutCase &cut_case : cut_cases )
        for( std::unique_ptr<CutOpWithNamesAndInds> &cownai : cut_case.possibilities )
            gggd.needed_cut_ops.insert( cownai->cut_op );

    // code for each case
    for( std::size_t num_cut_case = 0; num_cut_case < cut_cases.size(); ++num_cut_case ) {
        CutCase &cut_case = cut_cases[ num_cut_case ];
        for( std::size_t num_poss = 0; num_poss < cut_case.possibilities.size(); ++num_poss ) {
            std::unique_ptr<CutOpWithNamesAndInds> &cownai = cut_case.possibilities[ num_poss ];
            os << "    ks->" << cownai->cut_op.mk_item_func_name() << "( ";
            for( TI num_output = 0; num_output < cownai->outputs.size(); ++num_output ) {
                CutOpWithNamesAndInds::Out &output = cownai->outputs[ num_output ];
                os << "nsd_" << output.shape_name << ", {";
                for( TI n = 0; n < output.output_node_inds.size(); ++n )
                    os << ( n ? ", " : " " ) << output.output_node_inds[ n ];
                os << " }, {";
                for( TI n = 0; n < output.output_face_inds.size(); ++n )
                    os << ( n ? ", " : " " ) << output.output_face_inds[ n ];
                os << " }, ";
            }

            os << "old_shape_data, {";
            for( TI n = 0; n < cownai->input_node_inds.size(); ++n )
                os << ( n ? ", " : " " ) << cownai->input_node_inds[ n ];
            os << " }, {";
            for( TI n = 0; n < cownai->input_face_inds.size(); ++n )
                os << ( n ? ", " : " " ) << cownai->input_face_inds[ n ];
            os << " }, old_shape_data.cut_case_offsets[ " << num_cut_case << " ][ " << num_poss << " ], old_shape_data.cut_case_offsets[ " << num_cut_case << " ][ " << num_poss + 1 << " ], cut_ids, N<" << polytop.dim() << ">() );\n";
        }
    }

    os << "}\n";
}

void NamedRecursivePolytop::write_cut_cnt( std::ostream &os, std::vector<CutCase> &cut_cases ) const {
    os << "\n";
    os << "void " << name << "::cut_rese( const std::function<void(const ShapeType *,BI)> &fc, KernelSlot *ks, const ShapeData &sd ) const {\n";

    // type of produced shapes
    std::set<std::string> produced_shapes;
    for( const CutCase &cut_case : cut_cases )
        for( const std::unique_ptr<CutOpWithNamesAndInds> &cownai : cut_case.possibilities )
            for( const CutOpWithNamesAndInds::Out &out : cownai->outputs )
                produced_shapes.insert( out.shape_name );

    // there are cases with several possibilities
    bool need_sep_sub_cases = false;
    for( CutCase &cc : cut_cases )
        need_sep_sub_cases |= cc.possibilities.size() > 1;

    // reservation for score_best_sub_case and index_best_sub_case
    if ( need_sep_sub_cases ) {
        os << "    BI max_nb_item_with_sub_case = 0;\n";
        for( std::size_t n = 0; n < cut_cases.size(); ++n )
            if ( cut_cases[ n ].possibilities.size() > 1 )
                os << "    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ " << n << " ][ 1 ] - sd.cut_case_offsets[ " << n << " ][ 0 ] );\n";

        os << "\n";
        os << "    void *score_best_sub_case = ks->allocate_TF( max_nb_item_with_sub_case );\n";
        os << "    void *index_best_sub_case = ks->allocate_TI( max_nb_item_with_sub_case );\n";

        for( std::uint64_t n = 0; n < cut_cases.size(); ++n ) {
            CutCase &cc = cut_cases[ n ];
            if ( cc.possibilities.size() <= 1 )
                continue;

            // scores
            os << "\n";
            os << "    if ( sd.cut_case_offsets[ " << n << " ][ 1 ] - sd.cut_case_offsets[ " << n << " ][ 0 ] ) {\n";
            os << "        static std::vector<BI> nv{";
            std::vector<TI> off_scores;
            for( TI p = 0, off = 0; p < cut_cases[ n ].possibilities.size(); ++p ) {
                CutOpWithNamesAndInds &poss = *cut_cases[ n ].possibilities[ p ];
                for( const CutItem &cut_item : poss.cut_op.cut_items ) {
                    for( const std::array<std::array<TI,2>,2> &p : cut_item.lengths ) {
                        os << "\n            " << p[ 0 ][ 0 ] << ", " << p[ 0 ][ 1 ] << ", " << p[ 1 ][ 0 ] << ", " << p[ 1 ][ 1 ] << ",";
                        off += 4;
                    }
                }
                off_scores.push_back( off );
            }
            os << "\n        };\n";

            os << "\n";
            os << "        VecTI nn{ ks, nv };\n";
            os << "        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ " << n << " ][ 1 ] - sd.cut_case_offsets[ " << n << " ][ 0 ] );\n";

            os << "\n";
            for( std::size_t p = 0; p < cc.possibilities.size(); ++p ) {
                TI o = p ? off_scores[ p - 1 ] : 0;
                os << "        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ " << n << " ][ 0 ], sd.cut_case_offsets[ " << n << " ][ 1 ], " << p << ", nn.data(), "
                   << o << ", " << ( off_scores[ p ] - o ) / 4 << ", N<" << polytop.dim() << ">() );\n";
            }

            os << "        ks->sort_TI_in_range( sd.cut_case_offsets[ " << n << " ].data(), index_best_sub_case, sd.cut_case_offsets[ " << n << " ][ 1 ] - sd.cut_case_offsets[ " << n << " ][ 0 ], " << cc.possibilities.size() << ", sd.cut_indices, sd.cut_case_offsets[ " << n << " ][ 0 ] );\n";

            os << "    }\n";

            // sort by possibility number
        }
    }

    //
    for( std::string shape_name : produced_shapes ) {
        os << "\n";
        os << "    fc( s" << shape_name.substr( 1 ) << "(),";
        for( std::uint64_t n = 0, c = 0; n < cut_cases.size(); ++n )
            for( std::size_t p = 0; p < cut_cases[ n ].possibilities.size(); ++p )
                if ( std::size_t co = cut_cases[ n ].possibilities[ p ]->nb_created( shape_name ) )
                    os << ( c++ ? " +" : "" ) << "\n        ( sd.cut_case_offsets[ " << n << " ][ " << p + 1 << " ] - sd.cut_case_offsets[ " << n << " ][ " << p << " ] ) * " << co;
        os << "\n    );\n";
    }

    // free memory for score_best_sub_case and index_best_sub_case
    if ( need_sep_sub_cases ) {
        os << "    ks->free_TF( score_best_sub_case );\n";
        os << "    ks->free_TI( index_best_sub_case );\n";
    }

    os << "}\n";
}

void NamedRecursivePolytop::write_dsp_vtk( std::ostream &os ) const {
    std::string vtk_name = "polygon";
    if ( name == "S3" ) vtk_name = "triangle";
    if ( name == "S4" ) vtk_name = "quad";

    os << "\n";
    os << "void " << name << "::display_vtk( VtkOutput &vo, const double **tfs, const BI **tis, unsigned /*dim*/, BI nb_items, VtkOutput::Pt *offsets ) const {\n";
    os << "    using Pt = VtkOutput::Pt;\n";
    os << "    if ( offsets ) {\n";
    os << "        for( BI i = 0; i < nb_items; ++i ) {\n";
    os << "            vo.add_" << vtk_name << "( {\n";
    for( TI i = 0; i < polytop.points.size(); ++i )
        os << "                 Pt{ tfs[ " << 2 * i + 0 << " ][ i ], tfs[ " << 2 * i + 1 << " ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],\n";
    os << "            } );\n";
    os << "        }\n";
    os << "    } else {\n";
    os << "        for( BI i = 0; i < nb_items; ++i ) {\n";
    os << "            vo.add_" << vtk_name << "( {\n";
    for( TI i = 0; i < polytop.points.size(); ++i )
        os << "                 Pt{ tfs[ " << 2 * i + 0 << " ][ i ], tfs[ " << 2 * i + 1 << " ][ i ], 0.0 },\n";
    os << "            } );\n";
    os << "        }\n";
    os << "    }\n";
    os << "}\n";
}

}
