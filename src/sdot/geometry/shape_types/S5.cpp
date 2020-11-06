// generated file
#include "../ShapeData.h"
#include "../VtkOutput.h"
#include <iostream>
#include "S3.h"
#include "S4.h"
#include "S5.h"
#include "S6.h"

namespace parex {

class S5 : public ShapeType {
public:
    virtual std::vector<BI> cut_poss_count() const override;
    virtual void            display_vtk   ( VtkOutput &vo, const double **tfs, const BI **tis, unsigned dim, BI nb_items, VtkOutput::Pt *offsets ) const override;
    virtual void            cut_rese      ( const std::function<void(const ShapeType *,BI)> &fc, KernelSlot *ks, const ShapeData &sd ) const override;
    virtual unsigned        nb_nodes      () const override { return 5; }
    virtual unsigned        nb_faces      () const override { return 5; }
    virtual void            cut_ops       ( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const override;
    virtual std::string     name          () const override { return "S5"; }
};


std::vector<ShapeType::BI> S5::cut_poss_count() const {
    return { 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 0 };
}

void S5::cut_ops( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const {
    ShapeData &nsd_S3 = new_shape_map.find( s3() )->second;
    ShapeData &nsd_S4 = new_shape_map.find( s4() )->second;
    ShapeData &nsd_S5 = new_shape_map.find( s5() )->second;
    ShapeData &nsd_S6 = new_shape_map.find( s6() )->second;

    ks->mk_items_n5_0_0_1_1_2_2_3_3_4_4_f5_0_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 0 ][ 0 ], old_shape_data.cut_case_offsets[ 0 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_1_1_1_2_2_3_3_4_4_0_4_f6_0_1_2_3_4_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 1 ][ 0 ], old_shape_data.cut_case_offsets[ 1 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_0_0_1_1_2_2_2_3_3_4_4_f6_0_c_1_2_3_4( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 2 ][ 0 ], old_shape_data.cut_case_offsets[ 2 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n5_0_4_1_2_2_2_3_3_4_4_f5_c_1_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 3 ][ 0 ], old_shape_data.cut_case_offsets[ 3 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_0_1_1_1_2_2_3_3_3_4_4_f6_0_1_c_2_3_4( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 4 ][ 0 ], old_shape_data.cut_case_offsets[ 4 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_4_2_3_3_3_4_4_f4_c_2_3_4_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 5 ][ 0 ], old_shape_data.cut_case_offsets[ 5 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_1_2_2_3_f4_0_1_c_i_n5_0_1_2_3_3_3_4_4_0_4_f5_i_2_3_4_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 5 ][ 1 ], old_shape_data.cut_case_offsets[ 5 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n5_0_0_0_1_2_3_3_3_4_4_f5_0_c_2_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 6 ][ 0 ], old_shape_data.cut_case_offsets[ 6 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_4_2_3_3_3_4_4_f4_c_2_3_4( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 7 ][ 0 ], old_shape_data.cut_case_offsets[ 7 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_0_1_1_2_2_2_3_3_4_4_4_f6_0_1_2_c_3_4( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 8 ][ 0 ], old_shape_data.cut_case_offsets[ 8 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_4_3_4_4_4_f3_c_3_4_n4_0_1_1_1_2_2_2_3_f4_0_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 9 ][ 0 ], old_shape_data.cut_case_offsets[ 9 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_2_2_2_3_f4_0_1_2_i_n5_0_1_2_3_3_4_4_4_0_4_f5_i_c_3_4_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 9 ][ 1 ], old_shape_data.cut_case_offsets[ 9 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n4_0_0_0_1_3_4_4_4_f4_0_c_3_4( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 10 ][ 0 ], old_shape_data.cut_case_offsets[ 10 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_0_0_1_1_2_2_2_f4_0_c_1_i_n5_0_0_2_2_2_3_3_4_4_4_f5_i_2_c_3_4( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 10 ][ 1 ], old_shape_data.cut_case_offsets[ 10 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_4_3_4_4_4_f3_c_3_4_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 11 ][ 0 ], old_shape_data.cut_case_offsets[ 11 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_4_1_2_2_2_2_3_3_4_4_4_f6_c_1_2_c_3_4( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 11 ][ 1 ], old_shape_data.cut_case_offsets[ 11 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_1_2_3_4_4_4_f5_0_1_c_3_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 12 ][ 0 ], old_shape_data.cut_case_offsets[ 12 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_4_3_4_4_4_f3_c_3_4_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 13 ][ 0 ], old_shape_data.cut_case_offsets[ 13 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_1_1_1_1_2_3_4_4_4_0_4_f6_0_1_c_3_4_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 13 ][ 1 ], old_shape_data.cut_case_offsets[ 13 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_0_0_1_3_4_4_4_f4_0_c_3_4( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 14 ][ 0 ], old_shape_data.cut_case_offsets[ 14 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_4_3_4_4_4_f3_c_3_4( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 15 ][ 0 ], old_shape_data.cut_case_offsets[ 15 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_0_1_1_2_2_3_3_3_4_0_4_f6_0_1_2_3_c_4( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 16 ][ 0 ], old_shape_data.cut_case_offsets[ 16 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n5_0_1_1_1_2_2_3_3_3_4_f5_0_1_2_3_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 17 ][ 0 ], old_shape_data.cut_case_offsets[ 17 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_3_3_3_4_f4_1_2_3_c_n3_0_0_0_1_0_4_f3_0_c_4( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 18 ][ 0 ], old_shape_data.cut_case_offsets[ 18 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_0_0_1_1_2_2_2_f4_0_c_1_i_n5_0_0_2_2_3_3_3_4_0_4_f5_i_2_3_c_4( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 18 ][ 1 ], old_shape_data.cut_case_offsets[ 18 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_3_3_3_4_f4_1_2_3_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 19 ][ 0 ], old_shape_data.cut_case_offsets[ 19 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c_n4_0_0_1_1_1_2_0_4_f4_0_1_c_4( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 20 ][ 0 ], old_shape_data.cut_case_offsets[ 20 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_0_1_1_1_2_2_3_f4_0_1_c_i_n5_0_0_2_3_3_3_3_4_0_4_f5_i_2_3_c_4( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 20 ][ 1 ], old_shape_data.cut_case_offsets[ 20 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 21 ][ 0 ], old_shape_data.cut_case_offsets[ 21 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_1_1_1_1_2_2_3_3_3_3_4_f6_0_1_c_2_3_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 21 ][ 1 ], old_shape_data.cut_case_offsets[ 21 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c_n3_0_0_0_1_0_4_f3_0_c_4( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 22 ][ 0 ], old_shape_data.cut_case_offsets[ 22 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_0_0_1_2_3_3_3_3_4_0_4_f6_0_c_2_3_c_4( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 22 ][ 1 ], old_shape_data.cut_case_offsets[ 22 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 23 ][ 0 ], old_shape_data.cut_case_offsets[ 23 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n5_0_0_1_1_2_2_2_3_0_4_f5_0_1_2_c_4( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 24 ][ 0 ], old_shape_data.cut_case_offsets[ 24 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_2_2_2_3_f4_0_1_2_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2 }, old_shape_data.cut_case_offsets[ 25 ][ 0 ], old_shape_data.cut_case_offsets[ 25 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n3_0_0_0_1_0_4_f3_0_c_4( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 26 ][ 0 ], old_shape_data.cut_case_offsets[ 26 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_0_0_1_1_2_2_2_2_3_0_4_f6_0_c_1_2_c_4( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 26 ][ 1 ], old_shape_data.cut_case_offsets[ 26 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2 }, old_shape_data.cut_case_offsets[ 27 ][ 0 ], old_shape_data.cut_case_offsets[ 27 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_0_1_1_1_2_0_4_f4_0_1_c_4( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 28 ][ 0 ], old_shape_data.cut_case_offsets[ 28 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, { 0, 1 }, old_shape_data.cut_case_offsets[ 29 ][ 0 ], old_shape_data.cut_case_offsets[ 29 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_0_0_1_0_4_f3_0_c_4( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 30 ][ 0 ], old_shape_data.cut_case_offsets[ 30 ][ 1 ], cut_ids, N<2>() );
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

void S5::cut_rese( const std::function<void(const ShapeType *,BI)> &fc, KernelSlot *ks, const ShapeData &sd ) const {
    BI max_nb_item_with_sub_case = 0;
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 9 ][ 1 ] - sd.cut_case_offsets[ 9 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 11 ][ 1 ] - sd.cut_case_offsets[ 11 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 13 ][ 1 ] - sd.cut_case_offsets[ 13 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 18 ][ 1 ] - sd.cut_case_offsets[ 18 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 20 ][ 1 ] - sd.cut_case_offsets[ 20 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 21 ][ 1 ] - sd.cut_case_offsets[ 21 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 22 ][ 1 ] - sd.cut_case_offsets[ 22 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 26 ][ 1 ] - sd.cut_case_offsets[ 26 ][ 0 ] );

    void *score_best_sub_case = ks->allocate_TF( max_nb_item_with_sub_case );
    void *index_best_sub_case = ks->allocate_TI( max_nb_item_with_sub_case );

    if ( sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 4, 2, 3,
            1, 2, 0, 1,
            1, 2, 2, 3,
            0, 4, 0, 1,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 5 ][ 0 ], sd.cut_case_offsets[ 5 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 5 ][ 0 ], sd.cut_case_offsets[ 5 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );

        ks->sort_TI_in_range( sd.cut_case_offsets[ 5 ].data(), index_best_sub_case, sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ], 2, sd.cut_indices, sd.cut_case_offsets[ 5 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 9 ][ 1 ] - sd.cut_case_offsets[ 9 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 4, 3, 4,
            2, 3, 0, 1,
            2, 3, 3, 4,
            0, 4, 0, 1,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 9 ][ 1 ] - sd.cut_case_offsets[ 9 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 9 ][ 0 ], sd.cut_case_offsets[ 9 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 9 ][ 0 ], sd.cut_case_offsets[ 9 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );

        ks->sort_TI_in_range( sd.cut_case_offsets[ 9 ].data(), index_best_sub_case, sd.cut_case_offsets[ 9 ][ 1 ] - sd.cut_case_offsets[ 9 ][ 0 ], 2, sd.cut_indices, sd.cut_case_offsets[ 9 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] ) {
        static std::vector<BI> nv{
            2, 3, 1, 2,
            0, 1, 3, 4,
            0, 1, 1, 2,
            2, 3, 3, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 10 ][ 0 ], sd.cut_case_offsets[ 10 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 10 ][ 0 ], sd.cut_case_offsets[ 10 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );

        ks->sort_TI_in_range( sd.cut_case_offsets[ 10 ].data(), index_best_sub_case, sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ], 2, sd.cut_indices, sd.cut_case_offsets[ 10 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 11 ][ 1 ] - sd.cut_case_offsets[ 11 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 4, 3, 4,
            2, 3, 1, 2,
            0, 4, 1, 2,
            2, 3, 3, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 11 ][ 1 ] - sd.cut_case_offsets[ 11 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 11 ][ 0 ], sd.cut_case_offsets[ 11 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 11 ][ 0 ], sd.cut_case_offsets[ 11 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );

        ks->sort_TI_in_range( sd.cut_case_offsets[ 11 ].data(), index_best_sub_case, sd.cut_case_offsets[ 11 ][ 1 ] - sd.cut_case_offsets[ 11 ][ 0 ], 2, sd.cut_indices, sd.cut_case_offsets[ 11 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 13 ][ 1 ] - sd.cut_case_offsets[ 13 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 4, 3, 4,
            1, 2, 0, 1,
            1, 2, 3, 4,
            0, 4, 0, 1,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 13 ][ 1 ] - sd.cut_case_offsets[ 13 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 13 ][ 0 ], sd.cut_case_offsets[ 13 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 13 ][ 0 ], sd.cut_case_offsets[ 13 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );

        ks->sort_TI_in_range( sd.cut_case_offsets[ 13 ].data(), index_best_sub_case, sd.cut_case_offsets[ 13 ][ 1 ] - sd.cut_case_offsets[ 13 ][ 0 ], 2, sd.cut_indices, sd.cut_case_offsets[ 13 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 18 ][ 1 ] - sd.cut_case_offsets[ 18 ][ 0 ] ) {
        static std::vector<BI> nv{
            3, 4, 1, 2,
            0, 1, 0, 4,
            0, 1, 1, 2,
            3, 4, 0, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 18 ][ 1 ] - sd.cut_case_offsets[ 18 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 18 ][ 0 ], sd.cut_case_offsets[ 18 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 18 ][ 0 ], sd.cut_case_offsets[ 18 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );

        ks->sort_TI_in_range( sd.cut_case_offsets[ 18 ].data(), index_best_sub_case, sd.cut_case_offsets[ 18 ][ 1 ] - sd.cut_case_offsets[ 18 ][ 0 ], 2, sd.cut_indices, sd.cut_case_offsets[ 18 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 20 ][ 1 ] - sd.cut_case_offsets[ 20 ][ 0 ] ) {
        static std::vector<BI> nv{
            3, 4, 2, 3,
            1, 2, 0, 4,
            1, 2, 2, 3,
            3, 4, 0, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 20 ][ 1 ] - sd.cut_case_offsets[ 20 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 20 ][ 0 ], sd.cut_case_offsets[ 20 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 20 ][ 0 ], sd.cut_case_offsets[ 20 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );

        ks->sort_TI_in_range( sd.cut_case_offsets[ 20 ].data(), index_best_sub_case, sd.cut_case_offsets[ 20 ][ 1 ] - sd.cut_case_offsets[ 20 ][ 0 ], 2, sd.cut_indices, sd.cut_case_offsets[ 20 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 21 ][ 1 ] - sd.cut_case_offsets[ 21 ][ 0 ] ) {
        static std::vector<BI> nv{
            3, 4, 2, 3,
            1, 2, 0, 1,
            1, 2, 2, 3,
            3, 4, 0, 1,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 21 ][ 1 ] - sd.cut_case_offsets[ 21 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );

        ks->sort_TI_in_range( sd.cut_case_offsets[ 21 ].data(), index_best_sub_case, sd.cut_case_offsets[ 21 ][ 1 ] - sd.cut_case_offsets[ 21 ][ 0 ], 2, sd.cut_indices, sd.cut_case_offsets[ 21 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 22 ][ 1 ] - sd.cut_case_offsets[ 22 ][ 0 ] ) {
        static std::vector<BI> nv{
            3, 4, 2, 3,
            0, 1, 0, 4,
            0, 1, 2, 3,
            3, 4, 0, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 22 ][ 1 ] - sd.cut_case_offsets[ 22 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 22 ][ 0 ], sd.cut_case_offsets[ 22 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 22 ][ 0 ], sd.cut_case_offsets[ 22 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );

        ks->sort_TI_in_range( sd.cut_case_offsets[ 22 ].data(), index_best_sub_case, sd.cut_case_offsets[ 22 ][ 1 ] - sd.cut_case_offsets[ 22 ][ 0 ], 2, sd.cut_indices, sd.cut_case_offsets[ 22 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 26 ][ 1 ] - sd.cut_case_offsets[ 26 ][ 0 ] ) {
        static std::vector<BI> nv{
            2, 3, 1, 2,
            0, 1, 0, 4,
            0, 1, 1, 2,
            2, 3, 0, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 26 ][ 1 ] - sd.cut_case_offsets[ 26 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 26 ][ 0 ], sd.cut_case_offsets[ 26 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 26 ][ 0 ], sd.cut_case_offsets[ 26 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );

        ks->sort_TI_in_range( sd.cut_case_offsets[ 26 ].data(), index_best_sub_case, sd.cut_case_offsets[ 26 ][ 1 ] - sd.cut_case_offsets[ 26 ][ 0 ], 2, sd.cut_indices, sd.cut_case_offsets[ 26 ][ 0 ] );
    }

    fc( s3(),
        ( sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 9 ][ 1 ] - sd.cut_case_offsets[ 9 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 11 ][ 1 ] - sd.cut_case_offsets[ 11 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 13 ][ 1 ] - sd.cut_case_offsets[ 13 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 15 ][ 1 ] - sd.cut_case_offsets[ 15 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 18 ][ 1 ] - sd.cut_case_offsets[ 18 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 20 ][ 1 ] - sd.cut_case_offsets[ 20 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 1 ] - sd.cut_case_offsets[ 21 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 22 ][ 1 ] - sd.cut_case_offsets[ 22 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 23 ][ 1 ] - sd.cut_case_offsets[ 23 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 26 ][ 1 ] - sd.cut_case_offsets[ 26 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 27 ][ 1 ] - sd.cut_case_offsets[ 27 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 29 ][ 1 ] - sd.cut_case_offsets[ 29 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 30 ][ 1 ] - sd.cut_case_offsets[ 30 ][ 0 ] ) * 1
    );

    fc( s4(),
        ( sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 5 ][ 2 ] - sd.cut_case_offsets[ 5 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 7 ][ 1 ] - sd.cut_case_offsets[ 7 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 9 ][ 1 ] - sd.cut_case_offsets[ 9 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 9 ][ 2 ] - sd.cut_case_offsets[ 9 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 10 ][ 2 ] - sd.cut_case_offsets[ 10 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 14 ][ 1 ] - sd.cut_case_offsets[ 14 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 18 ][ 1 ] - sd.cut_case_offsets[ 18 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 18 ][ 2 ] - sd.cut_case_offsets[ 18 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 19 ][ 1 ] - sd.cut_case_offsets[ 19 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 20 ][ 1 ] - sd.cut_case_offsets[ 20 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 20 ][ 2 ] - sd.cut_case_offsets[ 20 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 25 ][ 1 ] - sd.cut_case_offsets[ 25 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 28 ][ 1 ] - sd.cut_case_offsets[ 28 ][ 0 ] ) * 1
    );

    fc( s5(),
        ( sd.cut_case_offsets[ 0 ][ 1 ] - sd.cut_case_offsets[ 0 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 3 ][ 1 ] - sd.cut_case_offsets[ 3 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 5 ][ 2 ] - sd.cut_case_offsets[ 5 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 6 ][ 1 ] - sd.cut_case_offsets[ 6 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 9 ][ 2 ] - sd.cut_case_offsets[ 9 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 10 ][ 2 ] - sd.cut_case_offsets[ 10 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 12 ][ 1 ] - sd.cut_case_offsets[ 12 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 17 ][ 1 ] - sd.cut_case_offsets[ 17 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 18 ][ 2 ] - sd.cut_case_offsets[ 18 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 20 ][ 2 ] - sd.cut_case_offsets[ 20 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 24 ][ 1 ] - sd.cut_case_offsets[ 24 ][ 0 ] ) * 1
    );

    fc( s6(),
        ( sd.cut_case_offsets[ 1 ][ 1 ] - sd.cut_case_offsets[ 1 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 2 ][ 1 ] - sd.cut_case_offsets[ 2 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 4 ][ 1 ] - sd.cut_case_offsets[ 4 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 8 ][ 1 ] - sd.cut_case_offsets[ 8 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 11 ][ 2 ] - sd.cut_case_offsets[ 11 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 13 ][ 2 ] - sd.cut_case_offsets[ 13 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 16 ][ 1 ] - sd.cut_case_offsets[ 16 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 2 ] - sd.cut_case_offsets[ 21 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 22 ][ 2 ] - sd.cut_case_offsets[ 22 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 26 ][ 2 ] - sd.cut_case_offsets[ 26 ][ 1 ] ) * 1
    );

    ks->free_TF( score_best_sub_case );
    ks->free_TI( index_best_sub_case );
}


// =======================================================================================
ShapeType *s5() {
    static S5 res;
    return &res;
}

} // namespace parex
