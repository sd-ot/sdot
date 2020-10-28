#include "../support/generic_ostream_output.h"
#include "../support/TODO.h"
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
    os << "#include \"../ShapeData.h\"\n";
    os << "#include \"../VtkOutput.h\"\n";
    for( const NamedRecursivePolytop &ps : available_primitive_shapes )
        os << "#include \"" << ps.name << ".h\"\n";
    os << "\n";

    os << "namespace sdot {\n";
    os << "\n";

    // class decl
    os << "class " << name << " : public ShapeType {\n";
    os << "public:\n";
    os << "    virtual void        display_vtk( VtkOutput &vo, const double **tfs, const BI **tis, unsigned dim, BI nb_items ) const override;\n";
    os << "    virtual void        cut_rese   ( const std::function<void(const ShapeType *,BI)> &fc, const BI *case_offsets ) const override;\n";
    os << "    virtual unsigned    nb_nodes   () const override { return " << polytop.points.size() << "; }\n";
    os << "    virtual unsigned    nb_faces   () const override { return " << polytop.nb_faces() << "; }\n";
    os << "    virtual void        cut_ops    ( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const override;\n";
    os << "    virtual std::string name       () const override { return \"" << name << "\"; }\n";
    os << "};\n";
    os << "\n";

    // definitions
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

void NamedRecursivePolytop::write_cut_ops( std::ostream &os, GlobGeneGeomData &gggd, std::vector<CutCase> &cut_cases ) const {
    os << "void " << name << "::cut_ops( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const {\n";

    // get (needed) output shape ptrs
    std::set<std::string> output_shape_names;
    for( CutCase &cut_case : cut_cases )
        for( CutOpWithNamesAndInds &cownai : cut_case.possibilities )
            for( CutOpWithNamesAndInds::Out &output : cownai.outputs )
                output_shape_names.insert( output.shape_name );
    for( std::string output_shape_name : output_shape_names )
        os << "    ShapeData &nsd_" << output_shape_name << " = new_shape_map.find( s" << output_shape_name.substr( 1 ) << "() )->second;\n";
    os << "\n";

    // reg needed functions in global generated code info
    for( CutCase &cut_case : cut_cases )
        for( CutOpWithNamesAndInds &cownai : cut_case.possibilities )
            gggd.needed_cut_ops.insert( cownai.cut_op );

    // code for each case
    for( std::size_t num_cut_case = 0; num_cut_case < cut_cases.size(); ++num_cut_case ) {
        CutCase &cut_case = cut_cases[ num_cut_case ];

        // nothing to create
        if ( cut_case.possibilities.empty() )
            continue;

        // one possibility => just call the corresponding function
        if ( cut_case.possibilities.size() == 1 ) {
            CutOpWithNamesAndInds &cownai = cut_case.possibilities[ 0 ];
            os << "    ks->" << cownai.cut_op.mk_item_func_name() << "( ";
            for( TI num_output = 0; num_output < cownai.outputs.size(); ++num_output ) {
                CutOpWithNamesAndInds::Out &output = cownai.outputs[ num_output ];
                os << "nsd_" << output.shape_name << ", {";
                for( TI n = 0; n < output.output_node_inds.size(); ++n )
                    os << ( n ? ", " : " " ) << output.output_node_inds[ n ];
                os << " }, {";
                for( TI n = 0; n < output.output_face_inds.size(); ++n )
                    os << ( n ? ", " : " " ) << output.output_face_inds[ n ];
                os << " }, ";
            }

            os << "old_shape_data, {";
            for( TI n = 0; n < cownai.input_node_inds.size(); ++n )
                os << ( n ? ", " : " " ) << cownai.input_node_inds[ n ];
            os << " }, {";
            for( TI n = 0; n < cownai.input_face_inds.size(); ++n )
                os << ( n ? ", " : " " ) << cownai.input_face_inds[ n ];
            os << " }, " << num_cut_case << ", cut_ids, N<" << polytop.dim() << ">() );\n";
            continue;
        }

        // several possibilities => need to get the best
        TODO;
    }

    os << "}\n";
    os << "\n";
}

void NamedRecursivePolytop::write_cut_cnt( std::ostream &os, std::vector<CutCase> &cut_cases ) const {
    // type of produced shapes
    std::set<std::string> produced_shapes;
    for( const CutCase &cut_case : cut_cases )
        for( const CutOpWithNamesAndInds &cownai : cut_case.possibilities )
            for( const CutOpWithNamesAndInds::Out &out : cownai.outputs )
                produced_shapes.insert( out.shape_name );

    //
    os << "void " << name << "::cut_rese( const std::function<void(const ShapeType *,BI)> &fc, const BI *case_offsets ) const {\n";
    for( std::string shape_name : produced_shapes ) {
        os << "    fc( s" << shape_name.substr( 1 ) << "(),";
        for( std::uint64_t n = 0, c = 0; n < cut_cases.size(); ++n ) {
            std::size_t v = 0;
            for( const CutOpWithNamesAndInds &possibilitie : cut_cases[ n ].possibilities )
                v = std::max( v, possibilitie.nb_created( name ) );
            if ( v ) {
                if ( c++ )
                    os << " +";
                os << "\n        ( case_offsets[ " << n + 1 << " ] - case_offsets[ " << n << " ] ) * " << v;
            }
        }
        os << "\n    );\n";
    }
    os << "}\n";
    os << "\n";
}

void NamedRecursivePolytop::write_dsp_vtk( std::ostream &os ) const {
    os << "void " << name << "::display_vtk( VtkOutput &vo, const double **tfs, const BI **/*tis*/, unsigned /*dim*/, BI nb_items ) const {\n";
    os << "    using Pt = VtkOutput::Pt;\n";
    os << "    for( BI i = 0; i < nb_items; ++i ) {\n";
    std::string vtk_name = "polygon";
    if ( name == "S3" ) vtk_name = "triangle";
    if ( name == "S4" ) vtk_name = "quad";
    os << "        vo.add_" << vtk_name << "( {\n";
    for( TI i = 0; i < polytop.points.size(); ++i )
        os << "             Pt{ tfs[ " << 2 * i + 0 << " ][ i ], tfs[ " << 2 * i + 1 << " ][ i ], 0.0 },\n";
    os << "        } );\n";
    os << "    }\n";
    os << "}\n";
    os << "\n";
}

}
