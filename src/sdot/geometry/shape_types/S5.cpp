// generated file
#include "../ShapeData.h"
#include "../VtkOutput.h"
#include "S3.h"
#include "S4.h"
#include "S5.h"

namespace sdot {

class S5 : public ShapeType {
public:
    virtual void        display_vtk( VtkOutput &vo, const double **tfs, const BI **tis, unsigned dim, BI nb_items, VtkOutput::Pt *offsets ) const override;
    virtual void        cut_rese   ( const std::function<void(const ShapeType *,BI)> &fc, const BI *cut_case_offsets ) const override;
    virtual unsigned    nb_nodes   () const override { return 5; }
    virtual unsigned    nb_faces   () const override { return 5; }
    virtual void        cut_ops    ( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const override;
    virtual std::string name       () const override { return "S5"; }
};

void S5::cut_ops( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const {
    ShapeData &nsd_S3 = new_shape_map.find( s3() )->second;
    ShapeData &nsd_S4 = new_shape_map.find( s4() )->second;
    ShapeData &nsd_S5 = new_shape_map.find( s5() )->second;

    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 0, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 2, cut_ids, N<2>() );
    ks->mk_items_n5_1_2_2_2_3_3_4_4_4_0_f5_1_2_3_4_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 3, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 4, cut_ids, N<2>() );
    ks->mk_items_n5_2_3_3_3_4_4_0_0_0_1_f5_2_3_4_0_c_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 6, cut_ids, N<2>() );
    ks->mk_items_n4_2_3_3_3_4_4_4_0_f4_2_3_4_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 7, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 8, cut_ids, N<2>() );
    ks->mk_items_n5_3_4_4_4_0_0_1_1_1_2_f5_3_4_0_1_c_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 12, cut_ids, N<2>() );
    ks->mk_items_n4_3_4_4_4_0_0_0_1_f4_3_4_0_c_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 14, cut_ids, N<2>() );
    ks->mk_items_n3_3_4_4_4_4_0_f3_3_4_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 15, cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 16, cut_ids, N<2>() );
    ks->mk_items_n5_0_1_1_1_2_2_3_3_3_4_f5_0_1_2_3_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, 17, cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_3_3_3_4_f4_1_2_3_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, 19, cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, 23, cut_ids, N<2>() );
    ks->mk_items_n5_4_0_0_0_1_1_2_2_2_3_f5_4_0_1_2_c_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 24, cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_2_2_2_3_f4_0_1_2_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2 }, 25, cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2 }, 27, cut_ids, N<2>() );
    ks->mk_items_n4_4_0_0_0_1_1_1_2_f4_4_0_1_c_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 28, cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, { 0, 1 }, 29, cut_ids, N<2>() );
    ks->mk_items_n3_4_0_0_0_0_1_f3_4_0_c_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, 30, cut_ids, N<2>() );
}

void S5::display_vtk( VtkOutput &vo, const double **tfs, const BI **tis, unsigned /*dim*/, BI nb_items, VtkOutput::Pt *offsets ) const {
    using Pt = VtkOutput::Pt;
    if ( offsets ) {
        for( BI i = 0; i < nb_items; ++i ) {
            vo.add_polygon( {
                 Pt{ tfs[ 0 ][ i ], tfs[ 1 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
                 Pt{ tfs[ 2 ][ i ], tfs[ 3 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
                 Pt{ tfs[ 4 ][ i ], tfs[ 5 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
                 Pt{ tfs[ 6 ][ i ], tfs[ 7 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
                 Pt{ tfs[ 8 ][ i ], tfs[ 9 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
            } );
        }
    } else {
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
}

void S5::cut_rese( const std::function<void(const ShapeType *,BI)> &fc, const BI *cut_case_offsets ) const {
    fc( s3(),
        ( cut_case_offsets[ 6 ] - cut_case_offsets[ 5 ] ) * 1 +
        ( cut_case_offsets[ 10 ] - cut_case_offsets[ 9 ] ) * 1 +
        ( cut_case_offsets[ 11 ] - cut_case_offsets[ 10 ] ) * 1 +
        ( cut_case_offsets[ 12 ] - cut_case_offsets[ 11 ] ) * 2 +
        ( cut_case_offsets[ 14 ] - cut_case_offsets[ 13 ] ) * 2 +
        ( cut_case_offsets[ 16 ] - cut_case_offsets[ 15 ] ) * 1 +
        ( cut_case_offsets[ 19 ] - cut_case_offsets[ 18 ] ) * 1 +
        ( cut_case_offsets[ 21 ] - cut_case_offsets[ 20 ] ) * 1 +
        ( cut_case_offsets[ 22 ] - cut_case_offsets[ 21 ] ) * 2 +
        ( cut_case_offsets[ 23 ] - cut_case_offsets[ 22 ] ) * 2 +
        ( cut_case_offsets[ 24 ] - cut_case_offsets[ 23 ] ) * 1 +
        ( cut_case_offsets[ 27 ] - cut_case_offsets[ 26 ] ) * 2 +
        ( cut_case_offsets[ 28 ] - cut_case_offsets[ 27 ] ) * 1 +
        ( cut_case_offsets[ 30 ] - cut_case_offsets[ 29 ] ) * 1 +
        ( cut_case_offsets[ 31 ] - cut_case_offsets[ 30 ] ) * 1
    );
    fc( s4(),
        ( cut_case_offsets[ 6 ] - cut_case_offsets[ 5 ] ) * 1 +
        ( cut_case_offsets[ 8 ] - cut_case_offsets[ 7 ] ) * 1 +
        ( cut_case_offsets[ 10 ] - cut_case_offsets[ 9 ] ) * 1 +
        ( cut_case_offsets[ 11 ] - cut_case_offsets[ 10 ] ) * 1 +
        ( cut_case_offsets[ 15 ] - cut_case_offsets[ 14 ] ) * 1 +
        ( cut_case_offsets[ 19 ] - cut_case_offsets[ 18 ] ) * 1 +
        ( cut_case_offsets[ 20 ] - cut_case_offsets[ 19 ] ) * 1 +
        ( cut_case_offsets[ 21 ] - cut_case_offsets[ 20 ] ) * 1 +
        ( cut_case_offsets[ 26 ] - cut_case_offsets[ 25 ] ) * 1 +
        ( cut_case_offsets[ 29 ] - cut_case_offsets[ 28 ] ) * 1
    );
    fc( s5(),
        ( cut_case_offsets[ 1 ] - cut_case_offsets[ 0 ] ) * 1 +
        ( cut_case_offsets[ 3 ] - cut_case_offsets[ 2 ] ) * 1 +
        ( cut_case_offsets[ 4 ] - cut_case_offsets[ 3 ] ) * 1 +
        ( cut_case_offsets[ 5 ] - cut_case_offsets[ 4 ] ) * 1 +
        ( cut_case_offsets[ 7 ] - cut_case_offsets[ 6 ] ) * 2 +
        ( cut_case_offsets[ 9 ] - cut_case_offsets[ 8 ] ) * 1 +
        ( cut_case_offsets[ 11 ] - cut_case_offsets[ 10 ] ) * 1 +
        ( cut_case_offsets[ 12 ] - cut_case_offsets[ 11 ] ) * 1 +
        ( cut_case_offsets[ 13 ] - cut_case_offsets[ 12 ] ) * 2 +
        ( cut_case_offsets[ 14 ] - cut_case_offsets[ 13 ] ) * 1 +
        ( cut_case_offsets[ 15 ] - cut_case_offsets[ 14 ] ) * 1 +
        ( cut_case_offsets[ 17 ] - cut_case_offsets[ 16 ] ) * 1 +
        ( cut_case_offsets[ 18 ] - cut_case_offsets[ 17 ] ) * 1 +
        ( cut_case_offsets[ 19 ] - cut_case_offsets[ 18 ] ) * 1 +
        ( cut_case_offsets[ 21 ] - cut_case_offsets[ 20 ] ) * 1 +
        ( cut_case_offsets[ 22 ] - cut_case_offsets[ 21 ] ) * 1 +
        ( cut_case_offsets[ 23 ] - cut_case_offsets[ 22 ] ) * 2 +
        ( cut_case_offsets[ 25 ] - cut_case_offsets[ 24 ] ) * 2 +
        ( cut_case_offsets[ 27 ] - cut_case_offsets[ 26 ] ) * 1 +
        ( cut_case_offsets[ 29 ] - cut_case_offsets[ 28 ] ) * 1 +
        ( cut_case_offsets[ 31 ] - cut_case_offsets[ 30 ] ) * 1
    );
}



// =======================================================================================
ShapeType *s5() {
    static S5 res;
    return &res;
}

} // namespace sdot
