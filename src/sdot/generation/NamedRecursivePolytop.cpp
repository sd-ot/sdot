#include "../support/generic_ostream_output.h"
#include "NamedRecursivePolytop.h"
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

void NamedRecursivePolytop::write_primitive_shape_impl( std::ostream &os, const std::vector<NamedRecursivePolytop> &/*available_primitive_shapes*/ ) const {
    std::vector<CutCase> cut_cases( std::uint64_t( 1 ) << polytop.points.size() );
    for( std::uint64_t n = 0; n < cut_cases.size(); ++n ) {
        std::vector<bool> outside( polytop.points.size() );
        for( std::uint64_t j = 0; j < polytop.points.size(); ++j )
            outside[ j ] = n & ( std::uint64_t( 1 ) << j );
        cut_cases[ n ].init( polytop, outside );
    }


    os << "// generated file\n";
    os << "#include \"../ShapeData.h\"\n";
    os << "#include \"../VtkOutput.h\"\n";
    os << "#include \"" << name << ".h\"\n";
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
    os << "        ShapeData &nsd = new_shape_map.find( this )->second;\n";
    os << "\n";
    os << "        ks->mk_items_0_0_1_1_2_2( nsd, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, 0, cut_ids, N<2>() );\n";
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
    os << "        ( offsets[ 1 ][ 0 ] - offsets[ 0 ][ 0 ] ) * 1 +\n";
    os << "        ( offsets[ 1 ][ 1 ] - offsets[ 0 ][ 1 ] ) * 0 +\n";
    os << "        ( offsets[ 1 ][ 2 ] - offsets[ 0 ][ 2 ] ) * 0 +\n";
    os << "        ( offsets[ 1 ][ 3 ] - offsets[ 0 ][ 3 ] ) * 0 +\n";
    os << "        ( offsets[ 1 ][ 4 ] - offsets[ 0 ][ 4 ] ) * 0 +\n";
    os << "        ( offsets[ 1 ][ 5 ] - offsets[ 0 ][ 5 ] ) * 0 +\n";
    os << "        ( offsets[ 1 ][ 6 ] - offsets[ 0 ][ 6 ] ) * 0 +\n";
    os << "        ( offsets[ 1 ][ 7 ] - offsets[ 0 ][ 7 ] ) * 0\n";
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

}
