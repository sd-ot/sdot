// generated file
#include "../ShapeData.h"
#include "../VtkOutput.h"
#include "S3.h"
#include "S4.h"
#include "S5.h"

namespace sdot {

class S4 : public ShapeType {
public:
    virtual void        display_vtk( VtkOutput &vo, const double **tfs, const BI **tis, unsigned dim, BI nb_items ) const override;
    virtual void        cut_rese   ( const std::function<void(const ShapeType *,BI)> &fc, const BI *case_offsets ) const override;
    virtual unsigned    nb_nodes   () const override { return 4; }
    virtual unsigned    nb_faces   () const override { return 4; }
    virtual void        cut_ops    ( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const override;
    virtual std::string name       () const override { return "S4"; }
};

void S4::cut_ops( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const {
    ShapeData &nsd_S4 = new_shape_map.find( s4() )->second;

    ks->mk_items_n4_0_0_1_1_2_2_3_3_f4_0_1_2_3( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, 0, cut_ids, N<2>() );
}

void S4::display_vtk( VtkOutput &vo, const double **tfs, const BI **/*tis*/, unsigned /*dim*/, BI nb_items ) const {
    using Pt = VtkOutput::Pt;
    for( BI i = 0; i < nb_items; ++i ) {
        vo.add_quad( {
             Pt{ tfs[ 0 ][ i ], tfs[ 1 ][ i ], 0.0 },
             Pt{ tfs[ 2 ][ i ], tfs[ 3 ][ i ], 0.0 },
             Pt{ tfs[ 4 ][ i ], tfs[ 5 ][ i ], 0.0 },
             Pt{ tfs[ 6 ][ i ], tfs[ 7 ][ i ], 0.0 },
        } );
    }
}

void S4::cut_rese( const std::function<void(const ShapeType *,BI)> &fc, const BI *case_offsets ) const {
    fc( s4(),
        ( case_offsets[ 1 ] - case_offsets[ 0 ] ) * 1
    );
}



// =======================================================================================
ShapeType *s4() {
    static S4 res;
    return &res;
}

} // namespace sdot
