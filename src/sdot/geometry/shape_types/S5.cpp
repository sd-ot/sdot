// generated file
#include "../ShapeData.h"
#include "../VtkOutput.h"
#include "S3.h"
#include "S4.h"
#include "S5.h"

namespace sdot {

class S5 : public ShapeType {
public:
    virtual void        display_vtk( VtkOutput &vo, const double **tfs, const BI **tis, unsigned dim, BI nb_items ) const override;
    virtual void        cut_rese   ( const std::function<void(const ShapeType *,BI)> &fc, const BI *case_offsets ) const override;
    virtual unsigned    nb_nodes   () const override { return 5; }
    virtual unsigned    nb_faces   () const override { return 5; }
    virtual void        cut_ops    ( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const override;
    virtual std::string name       () const override { return "S5"; }
};

void S5::cut_ops( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const {
    ShapeData &nsd_S5 = new_shape_map.find( s5() )->second;

    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 0, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 2, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 4, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 6, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 8, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 10, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 12, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 14, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 16, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 18, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 20, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 22, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 24, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 26, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 28, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 30, cut_ids, N<2>() );
}

void S5::display_vtk( VtkOutput &vo, const double **tfs, const BI **/*tis*/, unsigned /*dim*/, BI nb_items ) const {
    using Pt = VtkOutput::Pt;
    for( BI i = 0; i < nb_items; ++i ) {
        vo.add_polygon( {
             Pt{ tfs[ 0 ][ i ], tfs[ 1 ][ i ], 0.0 },
             Pt{ tfs[ 2 ][ i ], tfs[ 3 ][ i ], 0.0 },
             Pt{ tfs[ 4 ][ i ], tfs[ 5 ][ i ], 0.0 },
             Pt{ tfs[ 6 ][ i ], tfs[ 7 ][ i ], 0.0 },
             Pt{ tfs[ 8 ][ i ], tfs[ 9 ][ i ], 0.0 },
        } );
    }
}

void S5::cut_rese( const std::function<void(const ShapeType *,BI)> &fc, const BI *case_offsets ) const {
    fc( s5(),
        ( case_offsets[ 1 ] - case_offsets[ 0 ] ) * 1 +
        ( case_offsets[ 3 ] - case_offsets[ 2 ] ) * 1 +
        ( case_offsets[ 5 ] - case_offsets[ 4 ] ) * 1 +
        ( case_offsets[ 7 ] - case_offsets[ 6 ] ) * 1 +
        ( case_offsets[ 9 ] - case_offsets[ 8 ] ) * 1 +
        ( case_offsets[ 11 ] - case_offsets[ 10 ] ) * 1 +
        ( case_offsets[ 13 ] - case_offsets[ 12 ] ) * 1 +
        ( case_offsets[ 15 ] - case_offsets[ 14 ] ) * 1 +
        ( case_offsets[ 17 ] - case_offsets[ 16 ] ) * 1 +
        ( case_offsets[ 19 ] - case_offsets[ 18 ] ) * 1 +
        ( case_offsets[ 21 ] - case_offsets[ 20 ] ) * 1 +
        ( case_offsets[ 23 ] - case_offsets[ 22 ] ) * 1 +
        ( case_offsets[ 25 ] - case_offsets[ 24 ] ) * 1 +
        ( case_offsets[ 27 ] - case_offsets[ 26 ] ) * 1 +
        ( case_offsets[ 29 ] - case_offsets[ 28 ] ) * 1 +
        ( case_offsets[ 31 ] - case_offsets[ 30 ] ) * 1
    );
}



// =======================================================================================
ShapeType *s5() {
    static S5 res;
    return &res;
}

} // namespace sdot
