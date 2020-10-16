#include "../support/generic_ostream_output.h"
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
    os << "    virtual void        cut_count  ( const std::function<void(const ShapeType *,BI)> &fc, const BI **offsets ) const override;\n";
    os << "    virtual unsigned    nb_nodes   () const override { return " << polytop.points.size() << "; }\n";
    os << "    virtual unsigned    nb_faces   () const override { return " << polytop.nb_faces() << "; }\n";
    os << "    virtual std::string name       () const override { return \"" << name << "\"; }\n";
    os << "\n";

    // cut ops
    os << "    virtual void        cut_ops    ( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const override {\n";
    write_cut_ops( os, gggd, cut_cases );
    os << "    }\n";
    os << "};\n";
    os << "\n";

    // display_vtk
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

    // cut count
    os << "void " << name << "::cut_count( const std::function<void(const ShapeType *,BI)> &fc, const BI **offsets ) const {\n";
    os << "    fc( this,\n";
    for( std::uint64_t n = 0; n < cut_cases.size(); ++n )
        os << "        ( offsets[ 1 ][ " << n << " ] - offsets[ 0 ][ " << n << " ] ) * " << cut_cases[ n ].nb_created( name ) << ( n + 1 < cut_cases.size() ? " +" : "" ) << "\n";
    os << "    );\n";
    os << "}\n";
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
    // needed output shape data
    std::set<std::string> output_shape_names;
    for( CutCase &cut_case : cut_cases )
        for( CutOpWithNamesAndInds::Out &output : cut_case.cownai.outputs )
            output_shape_names.insert( output.shape_name );

    // get output shape ptrs
    for( std::string output_shape_name : output_shape_names )
        os << "        ShapeData &nsd_" << output_shape_name << " = new_shape_map.find( s" << output_shape_name.substr( 1 ) << "() )->second;\n";
    os << "\n";

    // reg in gggd
    for( CutCase &cut_case : cut_cases )
        gggd.needed_cut_ops.insert( cut_case.cownai.cut_op );

    // code for each case
    for( std::size_t num_cut_case = 0; num_cut_case < cut_cases.size(); ++num_cut_case ) {
        CutCase &cut_case = cut_cases[ num_cut_case ];
        if ( cut_case.cownai.cut_op ) {
            //
            os << "        ks->" << cut_case.cownai.cut_op.mk_item_func_name() << "( ";
            for( TI num_output = 0; num_output < cut_case.cownai.outputs.size(); ++num_output ) {
                CutOpWithNamesAndInds::Out &output = cut_case.cownai.outputs[ num_output ];
                os << "nsd_" << output.shape_name << ", {";
                for( TI n = 0; n < output.inds.size(); ++n )
                    os << ( n ? ", " : " " ) << output.inds[ n ];
                os << " }, ";
            }

            os << "old_shape_data, {";
            for( TI n = 0; n < cut_case.cownai.inputs.size(); ++n )
                os << ( n ? ", " : " " ) << cut_case.cownai.inputs[ n ];
            os << " }, " << num_cut_case << ", cut_ids, N<" << polytop.dim() << ">() );\n";
        }
    }
}

}
