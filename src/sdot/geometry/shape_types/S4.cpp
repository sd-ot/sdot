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
    ShapeData &nsd_S3 = new_shape_map.find( s3() )->second;
    ShapeData &nsd_S4 = new_shape_map.find( s4() )->second;
    ShapeData &nsd_S5 = new_shape_map.find( s5() )->second;

    ks->mk_items_n4_0_0_1_1_2_2_3_3_f4_0_1_2_3( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, 0, cut_ids, N<2>() );
    ks->mk_items_n5_0_1_1_1_2_2_3_3_3_0_f5_0_1_2_3_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, 1, cut_ids, N<2>() );
    ks->mk_items_n5_1_2_2_2_3_3_0_0_0_1_f5_1_2_3_0_c_n4_0_0_1_1_2_2_3_3_f4_0_1_2_3( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, 2, cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_3_3_3_0_f4_1_2_3_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, 3, cut_ids, N<2>() );
    ks->mk_items_n5_2_3_3_3_0_0_1_1_1_2_f5_2_3_0_1_c_n4_0_0_1_1_2_2_3_3_f4_0_1_2_3( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, 4, cut_ids, N<2>() );
    ks->mk_items_n4_2_3_3_3_0_0_0_1_f4_2_3_0_c_n4_0_0_1_1_2_2_3_3_f4_0_1_2_3( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, 6, cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_0_f3_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, 7, cut_ids, N<2>() );
    ks->mk_items_n5_3_0_0_0_1_1_2_2_2_3_f5_3_0_1_2_c_n4_0_0_1_1_2_2_3_3_f4_0_1_2_3( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, 8, cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_2_2_2_3_f4_0_1_2_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2 }, 9, cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2 }, 11, cut_ids, N<2>() );
    ks->mk_items_n4_3_0_0_0_1_1_1_2_f4_3_0_1_c_n4_0_0_1_1_2_2_3_3_f4_0_1_2_3( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, 12, cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, { 0, 1 }, 13, cut_ids, N<2>() );
    ks->mk_items_n3_3_0_0_0_0_1_f3_3_0_c_n4_0_0_1_1_2_2_3_3_f4_0_1_2_3( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, 14, cut_ids, N<2>() );
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
    fc( s3(),
        ( case_offsets[ 6 ] - case_offsets[ 5 ] ) * 2 +
        ( case_offsets[ 8 ] - case_offsets[ 7 ] ) * 1 +
        ( case_offsets[ 11 ] - case_offsets[ 10 ] ) * 2 +
        ( case_offsets[ 12 ] - case_offsets[ 11 ] ) * 1 +
        ( case_offsets[ 14 ] - case_offsets[ 13 ] ) * 1 +
        ( case_offsets[ 15 ] - case_offsets[ 14 ] ) * 1
    );
    fc( s4(),
        ( case_offsets[ 1 ] - case_offsets[ 0 ] ) * 1 +
        ( case_offsets[ 3 ] - case_offsets[ 2 ] ) * 1 +
        ( case_offsets[ 4 ] - case_offsets[ 3 ] ) * 1 +
        ( case_offsets[ 5 ] - case_offsets[ 4 ] ) * 1 +
        ( case_offsets[ 7 ] - case_offsets[ 6 ] ) * 2 +
        ( case_offsets[ 9 ] - case_offsets[ 8 ] ) * 1 +
        ( case_offsets[ 10 ] - case_offsets[ 9 ] ) * 1 +
        ( case_offsets[ 11 ] - case_offsets[ 10 ] ) * 1 +
        ( case_offsets[ 13 ] - case_offsets[ 12 ] ) * 2 +
        ( case_offsets[ 15 ] - case_offsets[ 14 ] ) * 1
    );
    fc( s5(),
        ( case_offsets[ 2 ] - case_offsets[ 1 ] ) * 1 +
        ( case_offsets[ 3 ] - case_offsets[ 2 ] ) * 1 +
        ( case_offsets[ 5 ] - case_offsets[ 4 ] ) * 1 +
        ( case_offsets[ 6 ] - case_offsets[ 5 ] ) * 1 +
        ( case_offsets[ 9 ] - case_offsets[ 8 ] ) * 1 +
        ( case_offsets[ 11 ] - case_offsets[ 10 ] ) * 1
    );
}



// =======================================================================================
ShapeType *s4() {
    static S4 res;
    return &res;
}

} // namespace sdot
