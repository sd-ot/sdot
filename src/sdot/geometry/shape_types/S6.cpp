// generated file
#include "../../kernels/VecTI.h"
#include "../ShapeData.h"
#include "../VtkOutput.h"
#include <iostream>
#include "S3.h"
#include "S4.h"
#include "S5.h"
#include "S6.h"

namespace sdot {

class S6 : public ShapeType {
public:
    virtual std::vector<BI> cut_poss_count() const override;
    virtual void            display_vtk   ( VtkOutput &vo, const double **tfs, const BI **tis, unsigned dim, BI nb_items, VtkOutput::Pt *offsets ) const override;
    virtual void            cut_rese      ( const std::function<void(const ShapeType *,BI)> &fc, KernelSlot *ks, const ShapeData &sd ) const override;
    virtual unsigned        nb_nodes      () const override { return 6; }
    virtual unsigned        nb_faces      () const override { return 6; }
    virtual void            cut_ops       ( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const override;
    virtual std::string     name          () const override { return "S6"; }
};


std::vector<ShapeType::BI> S6::cut_poss_count() const {
    return { 1, 1, 1, 1, 1, 4, 1, 1, 1, 4, 4, 4, 1, 4, 1, 1, 1, 4, 4, 4, 4, 27, 4, 4, 1, 4, 4, 4, 1, 4, 1, 1, 1, 1, 4, 1, 4, 4, 4, 1, 4, 4, 27, 4, 4, 4, 4, 1, 1, 1, 4, 1, 4, 4, 4, 1, 1, 1, 4, 1, 1, 1, 1, 0 };
}

void S6::cut_ops( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const {
    ShapeData &nsd_S3 = new_shape_map.find( s3() )->second;
    ShapeData &nsd_S4 = new_shape_map.find( s4() )->second;
    ShapeData &nsd_S5 = new_shape_map.find( s5() )->second;
    ShapeData &nsd_S6 = new_shape_map.find( s6() )->second;

    ks->mk_items_n6_0_0_1_1_2_2_3_3_4_4_5_5_f6_0_1_2_3_4_5( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 0 ][ 0 ], old_shape_data.cut_case_offsets[ 0 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_2_2_3_3_f4_0_1_2_i_n5_3_3_4_4_5_5_5_0_0_1_f5_3_4_5_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 1 ][ 0 ], old_shape_data.cut_case_offsets[ 1 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_3_3_4_4_f4_1_2_3_i_n5_4_4_5_5_0_0_0_1_1_2_f5_4_5_0_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 2 ][ 0 ], old_shape_data.cut_case_offsets[ 2 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_1_2_2_2_3_3_4_4_5_5_5_0_f6_1_2_3_4_5_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 3 ][ 0 ], old_shape_data.cut_case_offsets[ 3 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_2_3_3_3_4_4_5_5_f4_2_3_4_i_n5_5_5_0_0_1_1_1_2_2_3_f5_5_0_1_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 4 ][ 0 ], old_shape_data.cut_case_offsets[ 4 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n5_2_3_3_3_4_4_5_5_5_0_f5_2_3_4_5_c_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 5 ][ 0 ], old_shape_data.cut_case_offsets[ 5 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n5_0_1_1_1_1_2_2_3_3_3_f5_0_1_c_2_i_n5_3_3_4_4_5_5_5_0_0_1_f5_3_4_5_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 5 ][ 1 ], old_shape_data.cut_case_offsets[ 5 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n5_2_3_3_3_4_4_5_5_5_0_f5_2_3_4_5_i_n5_5_0_0_1_1_1_1_2_2_3_f5_c_0_1_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 5 ][ 2 ], old_shape_data.cut_case_offsets[ 5 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c_n5_2_3_3_3_4_4_5_5_5_0_f5_2_3_4_5_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 5 ][ 3 ], old_shape_data.cut_case_offsets[ 5 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n6_2_3_3_3_4_4_5_5_0_0_0_1_f6_2_3_4_5_0_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 6 ][ 0 ], old_shape_data.cut_case_offsets[ 6 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n5_2_3_3_3_4_4_5_5_5_0_f5_2_3_4_5_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 7 ][ 0 ], old_shape_data.cut_case_offsets[ 7 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_3_4_4_4_5_5_0_0_f4_3_4_5_i_n5_0_0_1_1_2_2_2_3_3_4_f5_0_1_2_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 8 ][ 0 ], old_shape_data.cut_case_offsets[ 8 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_3_4_4_4_5_5_5_0_f4_3_4_5_c_n4_0_1_1_1_2_2_2_3_f4_0_1_2_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 9 ][ 0 ], old_shape_data.cut_case_offsets[ 9 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n5_0_1_1_1_2_2_2_3_3_4_f5_0_1_2_c_i_n5_3_4_4_4_5_5_5_0_0_1_f5_3_4_5_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 9 ][ 1 ], old_shape_data.cut_case_offsets[ 9 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n5_3_4_4_4_5_5_5_0_0_1_f5_3_4_5_c_i_n5_0_1_1_1_2_2_2_3_3_4_f5_0_1_2_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 9 ][ 2 ], old_shape_data.cut_case_offsets[ 9 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_2_2_2_3_f4_0_1_2_c_n4_3_4_4_4_5_5_5_0_f4_3_4_5_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 9 ][ 3 ], old_shape_data.cut_case_offsets[ 9 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n5_3_4_4_4_5_5_0_0_0_1_f5_3_4_5_0_i_n5_0_1_1_2_2_2_2_3_3_4_f5_c_1_2_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 10 ][ 0 ], old_shape_data.cut_case_offsets[ 10 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n5_3_4_4_4_5_5_0_0_0_1_f5_3_4_5_0_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 10 ][ 1 ], old_shape_data.cut_case_offsets[ 10 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n5_3_4_4_4_5_5_0_0_0_1_f5_3_4_5_0_c_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 10 ][ 2 ], old_shape_data.cut_case_offsets[ 10 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n5_1_2_2_2_2_3_3_4_4_4_f5_1_2_c_3_i_n5_4_4_5_5_0_0_0_1_1_2_f5_4_5_0_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 10 ][ 3 ], old_shape_data.cut_case_offsets[ 10 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n4_3_4_4_4_5_5_5_0_f4_3_4_5_c_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 11 ][ 0 ], old_shape_data.cut_case_offsets[ 11 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_2_3_3_4_f4_1_2_c_i_n5_3_4_4_4_5_5_5_0_1_2_f5_3_4_5_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 11 ][ 1 ], old_shape_data.cut_case_offsets[ 11 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_3_4_4_4_5_5_5_0_f4_3_4_5_i_n5_5_0_1_2_2_2_2_3_3_4_f5_c_1_2_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 11 ][ 2 ], old_shape_data.cut_case_offsets[ 11 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n4_3_4_4_4_5_5_5_0_f4_3_4_5_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 11 ][ 3 ], old_shape_data.cut_case_offsets[ 11 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n6_3_4_4_4_5_5_0_0_1_1_1_2_f6_3_4_5_0_1_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 12 ][ 0 ], old_shape_data.cut_case_offsets[ 12 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_3_4_4_4_5_5_5_0_f4_3_4_5_c_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 13 ][ 0 ], old_shape_data.cut_case_offsets[ 13 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_1_2_3_4_f4_0_1_c_i_n5_3_4_4_4_5_5_5_0_0_1_f5_3_4_5_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 13 ][ 1 ], old_shape_data.cut_case_offsets[ 13 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_3_4_4_4_5_5_5_0_f4_3_4_5_i_n5_5_0_0_1_1_1_1_2_3_4_f5_c_0_1_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 13 ][ 2 ], old_shape_data.cut_case_offsets[ 13 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c_n4_3_4_4_4_5_5_5_0_f4_3_4_5_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 13 ][ 3 ], old_shape_data.cut_case_offsets[ 13 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n5_3_4_4_4_5_5_0_0_0_1_f5_3_4_5_0_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 14 ][ 0 ], old_shape_data.cut_case_offsets[ 14 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_3_4_4_4_5_5_5_0_f4_3_4_5_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 15 ][ 0 ], old_shape_data.cut_case_offsets[ 15 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_4_5_5_5_0_0_1_1_f4_4_5_0_i_n5_1_1_2_2_3_3_3_4_4_5_f5_1_2_3_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 16 ][ 0 ], old_shape_data.cut_case_offsets[ 16 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_4_5_5_5_5_0_f3_4_5_c_n5_0_1_1_1_2_2_3_3_3_4_f5_0_1_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 17 ][ 0 ], old_shape_data.cut_case_offsets[ 17 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n5_0_1_1_1_2_2_3_3_3_4_f5_0_1_2_3_i_n5_3_4_4_5_5_5_5_0_0_1_f5_c_4_5_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 17 ][ 1 ], old_shape_data.cut_case_offsets[ 17 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n5_4_5_5_5_5_0_0_1_1_1_f5_4_5_c_0_i_n5_1_1_2_2_3_3_3_4_4_5_f5_1_2_3_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 17 ][ 2 ], old_shape_data.cut_case_offsets[ 17 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n5_0_1_1_1_2_2_3_3_3_4_f5_0_1_2_3_c_n3_4_5_5_5_5_0_f3_4_5_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 17 ][ 3 ], old_shape_data.cut_case_offsets[ 17 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n5_4_5_5_5_0_0_0_1_1_2_f5_4_5_0_c_i_n5_1_2_2_2_3_3_3_4_4_5_f5_1_2_3_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 18 ][ 0 ], old_shape_data.cut_case_offsets[ 18 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_3_3_3_4_f4_1_2_3_c_n4_4_5_5_5_0_0_0_1_f4_4_5_0_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 18 ][ 1 ], old_shape_data.cut_case_offsets[ 18 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_4_5_5_5_0_0_0_1_f4_4_5_0_c_n4_1_2_2_2_3_3_3_4_f4_1_2_3_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 18 ][ 2 ], old_shape_data.cut_case_offsets[ 18 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n5_1_2_2_2_3_3_3_4_4_5_f5_1_2_3_c_i_n5_4_5_5_5_0_0_0_1_1_2_f5_4_5_0_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 18 ][ 3 ], old_shape_data.cut_case_offsets[ 18 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n3_4_5_5_5_5_0_f3_4_5_c_n4_1_2_2_2_3_3_3_4_f4_1_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 19 ][ 0 ], old_shape_data.cut_case_offsets[ 19 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_3_3_3_4_f4_1_2_3_i_n5_3_4_4_5_5_5_5_0_1_2_f5_c_4_5_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 19 ][ 1 ], old_shape_data.cut_case_offsets[ 19 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_4_5_5_5_5_0_1_2_f4_4_5_c_i_n5_1_2_2_2_3_3_3_4_4_5_f5_1_2_3_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 19 ][ 2 ], old_shape_data.cut_case_offsets[ 19 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_3_3_3_4_f4_1_2_3_c_n3_4_5_5_5_5_0_f3_4_5_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 19 ][ 3 ], old_shape_data.cut_case_offsets[ 19 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n5_4_5_5_5_0_0_1_1_1_2_f5_4_5_0_1_i_n5_1_2_2_3_3_3_3_4_4_5_f5_c_2_3_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 20 ][ 0 ], old_shape_data.cut_case_offsets[ 20 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c_n5_4_5_5_5_0_0_1_1_1_2_f5_4_5_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 20 ][ 1 ], old_shape_data.cut_case_offsets[ 20 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n5_4_5_5_5_0_0_1_1_1_2_f5_4_5_0_1_c_n3_2_3_3_3_3_4_f3_2_3_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 20 ][ 2 ], old_shape_data.cut_case_offsets[ 20 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n5_2_3_3_3_3_4_4_5_5_5_f5_2_3_c_4_i_n5_5_5_0_0_1_1_1_2_2_3_f5_5_0_1_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 20 ][ 3 ], old_shape_data.cut_case_offsets[ 20 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n3_4_5_5_5_5_0_f3_4_5_c_n3_2_3_3_3_3_4_f3_2_3_c_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 0 ], old_shape_data.cut_case_offsets[ 21 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_2_3_3_3_3_4_4_5_5_5_5_0_f6_2_3_c_4_5_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 1 ], old_shape_data.cut_case_offsets[ 21 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n6_4_5_5_5_5_0_2_3_3_3_3_4_f6_4_5_c_2_3_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 2 ], old_shape_data.cut_case_offsets[ 21 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c_n3_4_5_5_5_5_0_f3_4_5_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 3 ], old_shape_data.cut_case_offsets[ 21 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n5_0_1_1_1_1_2_2_3_3_3_f5_0_1_c_2_i_n6_3_3_3_4_4_5_5_5_5_0_0_1_f6_3_c_4_5_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 4 ], old_shape_data.cut_case_offsets[ 21 ][ 5 ], cut_ids, N<2>() );
    ks->mk_items_n3_4_5_5_5_5_0_f3_4_5_c_n6_0_1_1_1_1_2_2_3_3_3_3_4_f6_0_1_c_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 5 ], old_shape_data.cut_case_offsets[ 21 ][ 6 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_1_1_1_1_2_2_3_3_3_3_4_f6_0_1_c_2_3_c_n3_4_5_5_5_5_0_f3_4_5_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 6 ], old_shape_data.cut_case_offsets[ 21 ][ 7 ], cut_ids, N<2>() );
    ks->mk_items_n5_4_5_5_5_5_0_0_1_1_1_f5_4_5_c_0_i_n6_1_1_1_2_2_3_3_3_3_4_4_5_f6_1_c_2_3_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 7 ], old_shape_data.cut_case_offsets[ 21 ][ 8 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c_n6_0_1_1_1_1_2_4_5_5_5_5_0_f6_0_1_c_4_5_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 8 ], old_shape_data.cut_case_offsets[ 21 ][ 9 ], cut_ids, N<2>() );
    ks->mk_items_n3_4_5_5_5_5_0_f3_4_5_c_n6_2_3_3_3_3_4_0_1_1_1_1_2_f6_2_3_c_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 9 ], old_shape_data.cut_case_offsets[ 21 ][ 10 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c_n3_4_5_5_5_5_0_f3_4_5_c_n3_2_3_3_3_3_4_f3_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 10 ], old_shape_data.cut_case_offsets[ 21 ][ 11 ], cut_ids, N<2>() );
    ks->mk_items_n6_4_5_5_5_5_0_0_1_1_1_1_2_f6_4_5_c_0_1_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 11 ], old_shape_data.cut_case_offsets[ 21 ][ 12 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_1_1_1_1_2_4_5_5_5_5_0_f6_0_1_c_4_5_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 12 ], old_shape_data.cut_case_offsets[ 21 ][ 13 ], cut_ids, N<2>() );
    ks->mk_items_n3_4_5_5_5_5_0_f3_4_5_c_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 13 ], old_shape_data.cut_case_offsets[ 21 ][ 14 ], cut_ids, N<2>() );
    ks->mk_items_n5_2_3_3_3_3_4_4_5_5_5_f5_2_3_c_4_i_n6_5_5_5_0_0_1_1_1_1_2_2_3_f6_5_c_0_1_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 14 ], old_shape_data.cut_case_offsets[ 21 ][ 15 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c_n6_2_3_3_3_3_4_4_5_5_5_5_0_f6_2_3_c_4_5_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 15 ], old_shape_data.cut_case_offsets[ 21 ][ 16 ], cut_ids, N<2>() );
    ks->mk_items_n6_2_3_3_3_3_4_4_5_5_5_5_0_f6_2_3_c_4_5_c_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 16 ], old_shape_data.cut_case_offsets[ 21 ][ 17 ], cut_ids, N<2>() );
    ks->mk_items_n5_0_1_1_1_1_2_2_3_3_3_f5_0_1_c_2_i_n6_3_3_3_4_4_5_5_5_5_0_0_1_f6_3_c_4_5_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 17 ], old_shape_data.cut_case_offsets[ 21 ][ 18 ], cut_ids, N<2>() );
    ks->mk_items_n5_4_5_5_5_5_0_0_1_1_1_f5_4_5_c_0_i_n6_1_1_1_2_2_3_3_3_3_4_4_5_f6_1_c_2_3_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 18 ], old_shape_data.cut_case_offsets[ 21 ][ 19 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c_n6_4_5_5_5_5_0_0_1_1_1_1_2_f6_4_5_c_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 19 ], old_shape_data.cut_case_offsets[ 21 ][ 20 ], cut_ids, N<2>() );
    ks->mk_items_n6_4_5_5_5_5_0_0_1_1_1_1_2_f6_4_5_c_0_1_c_n3_2_3_3_3_3_4_f3_2_3_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 20 ], old_shape_data.cut_case_offsets[ 21 ][ 21 ], cut_ids, N<2>() );
    ks->mk_items_n5_2_3_3_3_3_4_4_5_5_5_f5_2_3_c_4_i_n6_5_5_5_0_0_1_1_1_1_2_2_3_f6_5_c_0_1_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 21 ], old_shape_data.cut_case_offsets[ 21 ][ 22 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c_n6_4_5_5_5_5_0_2_3_3_3_3_4_f6_4_5_c_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 22 ], old_shape_data.cut_case_offsets[ 21 ][ 23 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c_n3_0_1_1_1_1_2_f3_0_1_c_n3_4_5_5_5_5_0_f3_4_5_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 21 ][ 23 ], old_shape_data.cut_case_offsets[ 21 ][ 24 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_1_1_1_1_2_2_3_3_3_3_4_f6_0_1_c_2_3_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 21 ][ 24 ], old_shape_data.cut_case_offsets[ 21 ][ 25 ], cut_ids, N<2>() );
    ks->mk_items_n6_2_3_3_3_3_4_0_1_1_1_1_2_f6_2_3_c_0_1_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 21 ][ 25 ], old_shape_data.cut_case_offsets[ 21 ][ 26 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c_n3_2_3_3_3_3_4_f3_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 21 ][ 26 ], old_shape_data.cut_case_offsets[ 21 ][ 27 ], cut_ids, N<2>() );
    ks->mk_items_n4_4_5_5_5_0_0_0_1_f4_4_5_0_i_n5_0_1_2_3_3_3_3_4_4_5_f5_c_2_3_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 22 ][ 0 ], old_shape_data.cut_case_offsets[ 22 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c_n4_4_5_5_5_0_0_0_1_f4_4_5_0_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 22 ][ 1 ], old_shape_data.cut_case_offsets[ 22 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_4_5_5_5_0_0_0_1_f4_4_5_0_c_n3_2_3_3_3_3_4_f3_2_3_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 22 ][ 2 ], old_shape_data.cut_case_offsets[ 22 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n4_2_3_3_3_3_4_4_5_f4_2_3_c_i_n5_4_5_5_5_0_0_0_1_2_3_f5_4_5_0_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 22 ][ 3 ], old_shape_data.cut_case_offsets[ 22 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n3_4_5_5_5_5_0_f3_4_5_c_n3_2_3_3_3_3_4_f3_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 23 ][ 0 ], old_shape_data.cut_case_offsets[ 23 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_2_3_3_3_3_4_4_5_5_5_5_0_f6_2_3_c_4_5_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 23 ][ 1 ], old_shape_data.cut_case_offsets[ 23 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n6_4_5_5_5_5_0_2_3_3_3_3_4_f6_4_5_c_2_3_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 23 ][ 2 ], old_shape_data.cut_case_offsets[ 23 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c_n3_4_5_5_5_5_0_f3_4_5_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 23 ][ 3 ], old_shape_data.cut_case_offsets[ 23 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n6_4_5_5_5_0_0_1_1_2_2_2_3_f6_4_5_0_1_2_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 24 ][ 0 ], old_shape_data.cut_case_offsets[ 24 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_4_5_5_5_5_0_f3_4_5_c_n4_0_1_1_1_2_2_2_3_f4_0_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 25 ][ 0 ], old_shape_data.cut_case_offsets[ 25 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_2_2_2_3_f4_0_1_2_i_n5_2_3_4_5_5_5_5_0_0_1_f5_c_4_5_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 25 ][ 1 ], old_shape_data.cut_case_offsets[ 25 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_4_5_5_5_5_0_0_1_f4_4_5_c_i_n5_0_1_1_1_2_2_2_3_4_5_f5_0_1_2_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 25 ][ 2 ], old_shape_data.cut_case_offsets[ 25 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_2_2_2_3_f4_0_1_2_c_n3_4_5_5_5_5_0_f3_4_5_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 25 ][ 3 ], old_shape_data.cut_case_offsets[ 25 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n4_4_5_5_5_0_0_0_1_f4_4_5_0_i_n5_0_1_1_2_2_2_2_3_4_5_f5_c_1_2_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 26 ][ 0 ], old_shape_data.cut_case_offsets[ 26 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n4_4_5_5_5_0_0_0_1_f4_4_5_0_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 26 ][ 1 ], old_shape_data.cut_case_offsets[ 26 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_4_5_5_5_0_0_0_1_f4_4_5_0_c_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 26 ][ 2 ], old_shape_data.cut_case_offsets[ 26 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_2_3_4_5_f4_1_2_c_i_n5_4_5_5_5_0_0_0_1_1_2_f5_4_5_0_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 26 ][ 3 ], old_shape_data.cut_case_offsets[ 26 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n3_4_5_5_5_5_0_f3_4_5_c_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 27 ][ 0 ], old_shape_data.cut_case_offsets[ 27 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_1_2_2_2_2_3_4_5_5_5_5_0_f6_1_2_c_4_5_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 27 ][ 1 ], old_shape_data.cut_case_offsets[ 27 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n6_4_5_5_5_5_0_1_2_2_2_2_3_f6_4_5_c_1_2_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 27 ][ 2 ], old_shape_data.cut_case_offsets[ 27 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n3_4_5_5_5_5_0_f3_4_5_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 27 ][ 3 ], old_shape_data.cut_case_offsets[ 27 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n5_4_5_5_5_0_0_1_1_1_2_f5_4_5_0_1_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 28 ][ 0 ], old_shape_data.cut_case_offsets[ 28 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_4_5_5_5_5_0_f3_4_5_c_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 29 ][ 0 ], old_shape_data.cut_case_offsets[ 29 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_1_1_1_1_2_4_5_5_5_5_0_f6_0_1_c_4_5_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 29 ][ 1 ], old_shape_data.cut_case_offsets[ 29 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n6_4_5_5_5_5_0_0_1_1_1_1_2_f6_4_5_c_0_1_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 29 ][ 2 ], old_shape_data.cut_case_offsets[ 29 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c_n3_4_5_5_5_5_0_f3_4_5_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 29 ][ 3 ], old_shape_data.cut_case_offsets[ 29 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n4_4_5_5_5_0_0_0_1_f4_4_5_0_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 30 ][ 0 ], old_shape_data.cut_case_offsets[ 30 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_4_5_5_5_5_0_f3_4_5_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 31 ][ 0 ], old_shape_data.cut_case_offsets[ 31 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_5_0_0_0_1_1_2_2_f4_5_0_1_i_n5_2_2_3_3_4_4_4_5_5_0_f5_2_3_4_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 32 ][ 0 ], old_shape_data.cut_case_offsets[ 32 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_1_1_1_2_2_3_3_4_4_4_5_f6_0_1_2_3_4_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 33 ][ 0 ], old_shape_data.cut_case_offsets[ 33 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n5_5_0_0_0_0_1_1_2_2_2_f5_5_0_c_1_i_n5_2_2_3_3_4_4_4_5_5_0_f5_2_3_4_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 34 ][ 0 ], old_shape_data.cut_case_offsets[ 34 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n5_1_2_2_2_3_3_4_4_4_5_f5_1_2_3_4_c_n3_5_0_0_0_0_1_f3_5_0_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 34 ][ 1 ], old_shape_data.cut_case_offsets[ 34 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n3_5_0_0_0_0_1_f3_5_0_c_n5_1_2_2_2_3_3_4_4_4_5_f5_1_2_3_4_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 34 ][ 2 ], old_shape_data.cut_case_offsets[ 34 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n5_1_2_2_2_3_3_4_4_4_5_f5_1_2_3_4_i_n5_4_5_5_0_0_0_0_1_1_2_f5_c_5_0_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 34 ][ 3 ], old_shape_data.cut_case_offsets[ 34 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n5_1_2_2_2_3_3_4_4_4_5_f5_1_2_3_4_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 35 ][ 0 ], old_shape_data.cut_case_offsets[ 35 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n5_5_0_0_0_1_1_1_2_2_3_f5_5_0_1_c_i_n5_2_3_3_3_4_4_4_5_5_0_f5_2_3_4_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 36 ][ 0 ], old_shape_data.cut_case_offsets[ 36 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_2_3_3_3_4_4_4_5_f4_2_3_4_c_n4_5_0_0_0_1_1_1_2_f4_5_0_1_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 36 ][ 1 ], old_shape_data.cut_case_offsets[ 36 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_5_0_0_0_1_1_1_2_f4_5_0_1_c_n4_2_3_3_3_4_4_4_5_f4_2_3_4_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 36 ][ 2 ], old_shape_data.cut_case_offsets[ 36 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n5_2_3_3_3_4_4_4_5_5_0_f5_2_3_4_c_i_n5_5_0_0_0_1_1_1_2_2_3_f5_5_0_1_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 36 ][ 3 ], old_shape_data.cut_case_offsets[ 36 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n4_2_3_3_3_4_4_4_5_f4_2_3_4_c_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 37 ][ 0 ], old_shape_data.cut_case_offsets[ 37 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_1_2_2_3_f4_0_1_c_i_n5_2_3_3_3_4_4_4_5_0_1_f5_2_3_4_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 37 ][ 1 ], old_shape_data.cut_case_offsets[ 37 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_2_3_3_3_4_4_4_5_f4_2_3_4_i_n5_4_5_0_1_1_1_1_2_2_3_f5_c_0_1_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 37 ][ 2 ], old_shape_data.cut_case_offsets[ 37 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c_n4_2_3_3_3_4_4_4_5_f4_2_3_4_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 37 ][ 3 ], old_shape_data.cut_case_offsets[ 37 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n4_5_0_0_0_0_1_2_3_f4_5_0_c_i_n5_2_3_3_3_4_4_4_5_5_0_f5_2_3_4_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 38 ][ 0 ], old_shape_data.cut_case_offsets[ 38 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_2_3_3_3_4_4_4_5_f4_2_3_4_c_n3_5_0_0_0_0_1_f3_5_0_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 38 ][ 1 ], old_shape_data.cut_case_offsets[ 38 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n3_5_0_0_0_0_1_f3_5_0_c_n4_2_3_3_3_4_4_4_5_f4_2_3_4_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 38 ][ 2 ], old_shape_data.cut_case_offsets[ 38 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n4_2_3_3_3_4_4_4_5_f4_2_3_4_i_n5_4_5_5_0_0_0_0_1_2_3_f5_c_5_0_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 38 ][ 3 ], old_shape_data.cut_case_offsets[ 38 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n4_2_3_3_3_4_4_4_5_f4_2_3_4_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 39 ][ 0 ], old_shape_data.cut_case_offsets[ 39 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n5_5_0_0_0_1_1_2_2_2_3_f5_5_0_1_2_i_n5_2_3_3_4_4_4_4_5_5_0_f5_c_3_4_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 40 ][ 0 ], old_shape_data.cut_case_offsets[ 40 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_3_4_4_4_4_5_f3_3_4_c_n5_5_0_0_0_1_1_2_2_2_3_f5_5_0_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 40 ][ 1 ], old_shape_data.cut_case_offsets[ 40 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n5_5_0_0_0_1_1_2_2_2_3_f5_5_0_1_2_c_n3_3_4_4_4_4_5_f3_3_4_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 40 ][ 2 ], old_shape_data.cut_case_offsets[ 40 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n5_3_4_4_4_4_5_5_0_0_0_f5_3_4_c_5_i_n5_0_0_1_1_2_2_2_3_3_4_f5_0_1_2_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 40 ][ 3 ], old_shape_data.cut_case_offsets[ 40 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n3_3_4_4_4_4_5_f3_3_4_c_n4_0_1_1_1_2_2_2_3_f4_0_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 41 ][ 0 ], old_shape_data.cut_case_offsets[ 41 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_2_2_2_3_f4_0_1_2_i_n5_2_3_3_4_4_4_4_5_0_1_f5_c_3_4_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 41 ][ 1 ], old_shape_data.cut_case_offsets[ 41 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_3_4_4_4_4_5_0_1_f4_3_4_c_i_n5_0_1_1_1_2_2_2_3_3_4_f5_0_1_2_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 41 ][ 2 ], old_shape_data.cut_case_offsets[ 41 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_2_2_2_3_f4_0_1_2_c_n3_3_4_4_4_4_5_f3_3_4_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 41 ][ 3 ], old_shape_data.cut_case_offsets[ 41 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n5_5_0_0_0_0_1_1_2_2_2_f5_5_0_c_1_i_n6_2_2_2_3_3_4_4_4_4_5_5_0_f6_2_c_3_4_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 0 ], old_shape_data.cut_case_offsets[ 42 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_3_4_4_4_4_5_f3_3_4_c_n6_5_0_0_0_0_1_1_2_2_2_2_3_f6_5_0_c_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 1 ], old_shape_data.cut_case_offsets[ 42 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n6_5_0_0_0_0_1_1_2_2_2_2_3_f6_5_0_c_1_2_c_n3_3_4_4_4_4_5_f3_3_4_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 2 ], old_shape_data.cut_case_offsets[ 42 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n5_3_4_4_4_4_5_5_0_0_0_f5_3_4_c_5_i_n6_0_0_0_1_1_2_2_2_2_3_3_4_f6_0_c_1_2_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 3 ], old_shape_data.cut_case_offsets[ 42 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n6_5_0_0_0_0_1_3_4_4_4_4_5_f6_5_0_c_3_4_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 4 ], old_shape_data.cut_case_offsets[ 42 ][ 5 ], cut_ids, N<2>() );
    ks->mk_items_n3_3_4_4_4_4_5_f3_3_4_c_n3_1_2_2_2_2_3_f3_1_2_c_n3_5_0_0_0_0_1_f3_5_0_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 5 ], old_shape_data.cut_case_offsets[ 42 ][ 6 ], cut_ids, N<2>() );
    ks->mk_items_n6_1_2_2_2_2_3_3_4_4_4_4_5_f6_1_2_c_3_4_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 42 ][ 6 ], old_shape_data.cut_case_offsets[ 42 ][ 7 ], cut_ids, N<2>() );
    ks->mk_items_n6_3_4_4_4_4_5_1_2_2_2_2_3_f6_3_4_c_1_2_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 42 ][ 7 ], old_shape_data.cut_case_offsets[ 42 ][ 8 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n3_3_4_4_4_4_5_f3_3_4_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 42 ][ 8 ], old_shape_data.cut_case_offsets[ 42 ][ 9 ], cut_ids, N<2>() );
    ks->mk_items_n3_5_0_0_0_0_1_f3_5_0_c_n3_3_4_4_4_4_5_f3_3_4_c_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 9 ], old_shape_data.cut_case_offsets[ 42 ][ 10 ], cut_ids, N<2>() );
    ks->mk_items_n6_3_4_4_4_4_5_5_0_0_0_0_1_f6_3_4_c_5_0_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 10 ], old_shape_data.cut_case_offsets[ 42 ][ 11 ], cut_ids, N<2>() );
    ks->mk_items_n6_5_0_0_0_0_1_3_4_4_4_4_5_f6_5_0_c_3_4_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 11 ], old_shape_data.cut_case_offsets[ 42 ][ 12 ], cut_ids, N<2>() );
    ks->mk_items_n3_3_4_4_4_4_5_f3_3_4_c_n3_5_0_0_0_0_1_f3_5_0_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 12 ], old_shape_data.cut_case_offsets[ 42 ][ 13 ], cut_ids, N<2>() );
    ks->mk_items_n5_1_2_2_2_2_3_3_4_4_4_f5_1_2_c_3_i_n6_4_4_4_5_5_0_0_0_0_1_1_2_f6_4_c_5_0_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 13 ], old_shape_data.cut_case_offsets[ 42 ][ 14 ], cut_ids, N<2>() );
    ks->mk_items_n3_5_0_0_0_0_1_f3_5_0_c_n6_1_2_2_2_2_3_3_4_4_4_4_5_f6_1_2_c_3_4_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 14 ], old_shape_data.cut_case_offsets[ 42 ][ 15 ], cut_ids, N<2>() );
    ks->mk_items_n6_1_2_2_2_2_3_3_4_4_4_4_5_f6_1_2_c_3_4_c_n3_5_0_0_0_0_1_f3_5_0_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 15 ], old_shape_data.cut_case_offsets[ 42 ][ 16 ], cut_ids, N<2>() );
    ks->mk_items_n5_5_0_0_0_0_1_1_2_2_2_f5_5_0_c_1_i_n6_2_2_2_3_3_4_4_4_4_5_5_0_f6_2_c_3_4_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 16 ], old_shape_data.cut_case_offsets[ 42 ][ 17 ], cut_ids, N<2>() );
    ks->mk_items_n3_3_4_4_4_4_5_f3_3_4_c_n6_1_2_2_2_2_3_5_0_0_0_0_1_f6_1_2_c_5_0_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 17 ], old_shape_data.cut_case_offsets[ 42 ][ 18 ], cut_ids, N<2>() );
    ks->mk_items_n3_5_0_0_0_0_1_f3_5_0_c_n6_3_4_4_4_4_5_1_2_2_2_2_3_f6_3_4_c_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 18 ], old_shape_data.cut_case_offsets[ 42 ][ 19 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n3_5_0_0_0_0_1_f3_5_0_c_n3_3_4_4_4_4_5_f3_3_4_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 19 ], old_shape_data.cut_case_offsets[ 42 ][ 20 ], cut_ids, N<2>() );
    ks->mk_items_n6_5_0_0_0_0_1_1_2_2_2_2_3_f6_5_0_c_1_2_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 20 ], old_shape_data.cut_case_offsets[ 42 ][ 21 ], cut_ids, N<2>() );
    ks->mk_items_n6_1_2_2_2_2_3_5_0_0_0_0_1_f6_1_2_c_5_0_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 21 ], old_shape_data.cut_case_offsets[ 42 ][ 22 ], cut_ids, N<2>() );
    ks->mk_items_n3_5_0_0_0_0_1_f3_5_0_c_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 22 ], old_shape_data.cut_case_offsets[ 42 ][ 23 ], cut_ids, N<2>() );
    ks->mk_items_n5_3_4_4_4_4_5_5_0_0_0_f5_3_4_c_5_i_n6_0_0_0_1_1_2_2_2_2_3_3_4_f6_0_c_1_2_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 23 ], old_shape_data.cut_case_offsets[ 42 ][ 24 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n6_3_4_4_4_4_5_5_0_0_0_0_1_f6_3_4_c_5_0_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 24 ], old_shape_data.cut_case_offsets[ 42 ][ 25 ], cut_ids, N<2>() );
    ks->mk_items_n6_3_4_4_4_4_5_5_0_0_0_0_1_f6_3_4_c_5_0_c_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 25 ], old_shape_data.cut_case_offsets[ 42 ][ 26 ], cut_ids, N<2>() );
    ks->mk_items_n5_1_2_2_2_2_3_3_4_4_4_f5_1_2_c_3_i_n6_4_4_4_5_5_0_0_0_0_1_1_2_f6_4_c_5_0_c_i( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 42 ][ 26 ], old_shape_data.cut_case_offsets[ 42 ][ 27 ], cut_ids, N<2>() );
    ks->mk_items_n3_3_4_4_4_4_5_f3_3_4_c_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 43 ][ 0 ], old_shape_data.cut_case_offsets[ 43 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_1_2_2_2_2_3_3_4_4_4_4_5_f6_1_2_c_3_4_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 43 ][ 1 ], old_shape_data.cut_case_offsets[ 43 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n6_3_4_4_4_4_5_1_2_2_2_2_3_f6_3_4_c_1_2_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 43 ][ 2 ], old_shape_data.cut_case_offsets[ 43 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n3_3_4_4_4_4_5_f3_3_4_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 43 ][ 3 ], old_shape_data.cut_case_offsets[ 43 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n4_5_0_0_0_1_1_1_2_f4_5_0_1_i_n5_1_2_3_4_4_4_4_5_5_0_f5_c_3_4_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 44 ][ 0 ], old_shape_data.cut_case_offsets[ 44 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_3_4_4_4_4_5_f3_3_4_c_n4_5_0_0_0_1_1_1_2_f4_5_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 44 ][ 1 ], old_shape_data.cut_case_offsets[ 44 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_5_0_0_0_1_1_1_2_f4_5_0_1_c_n3_3_4_4_4_4_5_f3_3_4_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 44 ][ 2 ], old_shape_data.cut_case_offsets[ 44 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n4_3_4_4_4_4_5_5_0_f4_3_4_c_i_n5_5_0_0_0_1_1_1_2_3_4_f5_5_0_1_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 44 ][ 3 ], old_shape_data.cut_case_offsets[ 44 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n3_3_4_4_4_4_5_f3_3_4_c_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 45 ][ 0 ], old_shape_data.cut_case_offsets[ 45 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_1_1_1_1_2_3_4_4_4_4_5_f6_0_1_c_3_4_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 45 ][ 1 ], old_shape_data.cut_case_offsets[ 45 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n6_3_4_4_4_4_5_0_1_1_1_1_2_f6_3_4_c_0_1_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 45 ][ 2 ], old_shape_data.cut_case_offsets[ 45 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c_n3_3_4_4_4_4_5_f3_3_4_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 45 ][ 3 ], old_shape_data.cut_case_offsets[ 45 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n6_5_0_0_0_0_1_3_4_4_4_4_5_f6_5_0_c_3_4_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 46 ][ 0 ], old_shape_data.cut_case_offsets[ 46 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_3_4_4_4_4_5_f3_3_4_c_n3_5_0_0_0_0_1_f3_5_0_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 46 ][ 1 ], old_shape_data.cut_case_offsets[ 46 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n3_5_0_0_0_0_1_f3_5_0_c_n3_3_4_4_4_4_5_f3_3_4_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 46 ][ 2 ], old_shape_data.cut_case_offsets[ 46 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n6_3_4_4_4_4_5_5_0_0_0_0_1_f6_3_4_c_5_0_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 46 ][ 3 ], old_shape_data.cut_case_offsets[ 46 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n3_3_4_4_4_4_5_f3_3_4_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4 }, old_shape_data.cut_case_offsets[ 47 ][ 0 ], old_shape_data.cut_case_offsets[ 47 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_5_0_0_0_1_1_2_2_3_3_3_4_f6_5_0_1_2_3_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 48 ][ 0 ], old_shape_data.cut_case_offsets[ 48 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n5_0_1_1_1_2_2_3_3_3_4_f5_0_1_2_3_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 49 ][ 0 ], old_shape_data.cut_case_offsets[ 49 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_5_0_0_0_0_1_1_2_f4_5_0_c_i_n5_1_2_2_2_3_3_3_4_5_0_f5_1_2_3_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 50 ][ 0 ], old_shape_data.cut_case_offsets[ 50 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_3_3_3_4_f4_1_2_3_c_n3_5_0_0_0_0_1_f3_5_0_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 50 ][ 1 ], old_shape_data.cut_case_offsets[ 50 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n3_5_0_0_0_0_1_f3_5_0_c_n4_1_2_2_2_3_3_3_4_f4_1_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 50 ][ 2 ], old_shape_data.cut_case_offsets[ 50 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_3_3_3_4_f4_1_2_3_i_n5_3_4_5_0_0_0_0_1_1_2_f5_c_5_0_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 50 ][ 3 ], old_shape_data.cut_case_offsets[ 50 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_3_3_3_4_f4_1_2_3_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 51 ][ 0 ], old_shape_data.cut_case_offsets[ 51 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_5_0_0_0_1_1_1_2_f4_5_0_1_i_n5_1_2_2_3_3_3_3_4_5_0_f5_c_2_3_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 52 ][ 0 ], old_shape_data.cut_case_offsets[ 52 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c_n4_5_0_0_0_1_1_1_2_f4_5_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 52 ][ 1 ], old_shape_data.cut_case_offsets[ 52 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_5_0_0_0_1_1_1_2_f4_5_0_1_c_n3_2_3_3_3_3_4_f3_2_3_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 52 ][ 2 ], old_shape_data.cut_case_offsets[ 52 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n4_2_3_3_3_3_4_5_0_f4_2_3_c_i_n5_5_0_0_0_1_1_1_2_2_3_f5_5_0_1_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 52 ][ 3 ], old_shape_data.cut_case_offsets[ 52 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 53 ][ 0 ], old_shape_data.cut_case_offsets[ 53 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_0_1_1_1_1_2_2_3_3_3_3_4_f6_0_1_c_2_3_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 53 ][ 1 ], old_shape_data.cut_case_offsets[ 53 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n6_2_3_3_3_3_4_0_1_1_1_1_2_f6_2_3_c_0_1_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 53 ][ 2 ], old_shape_data.cut_case_offsets[ 53 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c_n3_2_3_3_3_3_4_f3_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 53 ][ 3 ], old_shape_data.cut_case_offsets[ 53 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n6_5_0_0_0_0_1_2_3_3_3_3_4_f6_5_0_c_2_3_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 54 ][ 0 ], old_shape_data.cut_case_offsets[ 54 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c_n3_5_0_0_0_0_1_f3_5_0_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 54 ][ 1 ], old_shape_data.cut_case_offsets[ 54 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n3_5_0_0_0_0_1_f3_5_0_c_n3_2_3_3_3_3_4_f3_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 54 ][ 2 ], old_shape_data.cut_case_offsets[ 54 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n6_2_3_3_3_3_4_5_0_0_0_0_1_f6_2_3_c_5_0_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 54 ][ 3 ], old_shape_data.cut_case_offsets[ 54 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_4_f3_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 55 ][ 0 ], old_shape_data.cut_case_offsets[ 55 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n5_5_0_0_0_1_1_2_2_2_3_f5_5_0_1_2_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 56 ][ 0 ], old_shape_data.cut_case_offsets[ 56 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_2_2_2_3_f4_0_1_2_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2 }, old_shape_data.cut_case_offsets[ 57 ][ 0 ], old_shape_data.cut_case_offsets[ 57 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n6_5_0_0_0_0_1_1_2_2_2_2_3_f6_5_0_c_1_2_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 58 ][ 0 ], old_shape_data.cut_case_offsets[ 58 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n3_5_0_0_0_0_1_f3_5_0_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 58 ][ 1 ], old_shape_data.cut_case_offsets[ 58 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n3_5_0_0_0_0_1_f3_5_0_c_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 58 ][ 2 ], old_shape_data.cut_case_offsets[ 58 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n6_1_2_2_2_2_3_5_0_0_0_0_1_f6_1_2_c_5_0_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 58 ][ 3 ], old_shape_data.cut_case_offsets[ 58 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2 }, old_shape_data.cut_case_offsets[ 59 ][ 0 ], old_shape_data.cut_case_offsets[ 59 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_5_0_0_0_1_1_1_2_f4_5_0_1_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 60 ][ 0 ], old_shape_data.cut_case_offsets[ 60 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, { 0, 1 }, old_shape_data.cut_case_offsets[ 61 ][ 0 ], old_shape_data.cut_case_offsets[ 61 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_5_0_0_0_0_1_f3_5_0_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data.cut_case_offsets[ 62 ][ 0 ], old_shape_data.cut_case_offsets[ 62 ][ 1 ], cut_ids, N<2>() );
}

void S6::display_vtk( VtkOutput &vo, const double **tfs, const BI **tis, unsigned /*dim*/, BI nb_items, VtkOutput::Pt *offsets ) const {
    using Pt = VtkOutput::Pt;
    if ( offsets ) {
        for( BI i = 0; i < nb_items; ++i ) {
            vo.add_polygon( {
                 Pt{ tfs[ 0 ][ i ], tfs[ 1 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
                 Pt{ tfs[ 2 ][ i ], tfs[ 3 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
                 Pt{ tfs[ 4 ][ i ], tfs[ 5 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
                 Pt{ tfs[ 6 ][ i ], tfs[ 7 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
                 Pt{ tfs[ 8 ][ i ], tfs[ 9 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
                 Pt{ tfs[ 10 ][ i ], tfs[ 11 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
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
                 Pt{ tfs[ 10 ][ i ], tfs[ 11 ][ i ], 0.0 },
            } );
        }
    }
}

void S6::cut_rese( const std::function<void(const ShapeType *,BI)> &fc, KernelSlot *ks, const ShapeData &sd ) const {
    BI max_nb_item_with_sub_case = 0;
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 9 ][ 1 ] - sd.cut_case_offsets[ 9 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 11 ][ 1 ] - sd.cut_case_offsets[ 11 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 13 ][ 1 ] - sd.cut_case_offsets[ 13 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 17 ][ 1 ] - sd.cut_case_offsets[ 17 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 18 ][ 1 ] - sd.cut_case_offsets[ 18 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 19 ][ 1 ] - sd.cut_case_offsets[ 19 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 20 ][ 1 ] - sd.cut_case_offsets[ 20 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 21 ][ 1 ] - sd.cut_case_offsets[ 21 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 22 ][ 1 ] - sd.cut_case_offsets[ 22 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 23 ][ 1 ] - sd.cut_case_offsets[ 23 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 25 ][ 1 ] - sd.cut_case_offsets[ 25 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 26 ][ 1 ] - sd.cut_case_offsets[ 26 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 27 ][ 1 ] - sd.cut_case_offsets[ 27 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 29 ][ 1 ] - sd.cut_case_offsets[ 29 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 34 ][ 1 ] - sd.cut_case_offsets[ 34 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 36 ][ 1 ] - sd.cut_case_offsets[ 36 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 37 ][ 1 ] - sd.cut_case_offsets[ 37 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 38 ][ 1 ] - sd.cut_case_offsets[ 38 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 40 ][ 1 ] - sd.cut_case_offsets[ 40 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 41 ][ 1 ] - sd.cut_case_offsets[ 41 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 42 ][ 1 ] - sd.cut_case_offsets[ 42 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 43 ][ 1 ] - sd.cut_case_offsets[ 43 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 44 ][ 1 ] - sd.cut_case_offsets[ 44 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 45 ][ 1 ] - sd.cut_case_offsets[ 45 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 46 ][ 1 ] - sd.cut_case_offsets[ 46 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 50 ][ 1 ] - sd.cut_case_offsets[ 50 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 52 ][ 1 ] - sd.cut_case_offsets[ 52 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 53 ][ 1 ] - sd.cut_case_offsets[ 53 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 54 ][ 1 ] - sd.cut_case_offsets[ 54 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 58 ][ 1 ] - sd.cut_case_offsets[ 58 ][ 0 ] );

    void *score_best_sub_case = ks->allocate_TF( max_nb_item_with_sub_case );
    void *index_best_sub_case = ks->allocate_TI( max_nb_item_with_sub_case );

    if ( sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] ) {
        static std::vector<BI> nv{
            5, 0, 2, 3,
            1, 2, 0, 1,
            1, 2, 2, 3,
            5, 0, 0, 1,
            5, 0, 0, 1,
            1, 2, 2, 3,
            1, 2, 0, 1,
            5, 0, 2, 3,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 5 ][ 0 ], sd.cut_case_offsets[ 5 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 5 ][ 0 ], sd.cut_case_offsets[ 5 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 5 ][ 0 ], sd.cut_case_offsets[ 5 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 5 ][ 0 ], sd.cut_case_offsets[ 5 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 5 ].data(), index_best_sub_case, sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 5 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 9 ][ 1 ] - sd.cut_case_offsets[ 9 ][ 0 ] ) {
        static std::vector<BI> nv{
            5, 0, 3, 4,
            2, 3, 0, 1,
            2, 3, 3, 4,
            5, 0, 0, 1,
            5, 0, 0, 1,
            2, 3, 3, 4,
            2, 3, 0, 1,
            5, 0, 3, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 9 ][ 1 ] - sd.cut_case_offsets[ 9 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 9 ][ 0 ], sd.cut_case_offsets[ 9 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 9 ][ 0 ], sd.cut_case_offsets[ 9 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 9 ][ 0 ], sd.cut_case_offsets[ 9 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 9 ][ 0 ], sd.cut_case_offsets[ 9 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 9 ].data(), index_best_sub_case, sd.cut_case_offsets[ 9 ][ 1 ] - sd.cut_case_offsets[ 9 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 9 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 1, 1, 2,
            2, 3, 3, 4,
            2, 3, 1, 2,
            0, 1, 3, 4,
            0, 1, 3, 4,
            2, 3, 1, 2,
            2, 3, 3, 4,
            0, 1, 1, 2,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 10 ][ 0 ], sd.cut_case_offsets[ 10 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 10 ][ 0 ], sd.cut_case_offsets[ 10 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 10 ][ 0 ], sd.cut_case_offsets[ 10 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 10 ][ 0 ], sd.cut_case_offsets[ 10 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 10 ].data(), index_best_sub_case, sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 10 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 11 ][ 1 ] - sd.cut_case_offsets[ 11 ][ 0 ] ) {
        static std::vector<BI> nv{
            5, 0, 3, 4,
            2, 3, 1, 2,
            2, 3, 3, 4,
            5, 0, 1, 2,
            5, 0, 1, 2,
            2, 3, 3, 4,
            2, 3, 1, 2,
            5, 0, 3, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 11 ][ 1 ] - sd.cut_case_offsets[ 11 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 11 ][ 0 ], sd.cut_case_offsets[ 11 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 11 ][ 0 ], sd.cut_case_offsets[ 11 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 11 ][ 0 ], sd.cut_case_offsets[ 11 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 11 ][ 0 ], sd.cut_case_offsets[ 11 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 11 ].data(), index_best_sub_case, sd.cut_case_offsets[ 11 ][ 1 ] - sd.cut_case_offsets[ 11 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 11 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 13 ][ 1 ] - sd.cut_case_offsets[ 13 ][ 0 ] ) {
        static std::vector<BI> nv{
            5, 0, 3, 4,
            1, 2, 0, 1,
            1, 2, 3, 4,
            5, 0, 0, 1,
            5, 0, 0, 1,
            1, 2, 3, 4,
            1, 2, 0, 1,
            5, 0, 3, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 13 ][ 1 ] - sd.cut_case_offsets[ 13 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 13 ][ 0 ], sd.cut_case_offsets[ 13 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 13 ][ 0 ], sd.cut_case_offsets[ 13 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 13 ][ 0 ], sd.cut_case_offsets[ 13 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 13 ][ 0 ], sd.cut_case_offsets[ 13 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 13 ].data(), index_best_sub_case, sd.cut_case_offsets[ 13 ][ 1 ] - sd.cut_case_offsets[ 13 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 13 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 17 ][ 1 ] - sd.cut_case_offsets[ 17 ][ 0 ] ) {
        static std::vector<BI> nv{
            5, 0, 4, 5,
            3, 4, 0, 1,
            3, 4, 4, 5,
            5, 0, 0, 1,
            5, 0, 0, 1,
            3, 4, 4, 5,
            3, 4, 0, 1,
            5, 0, 4, 5,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 17 ][ 1 ] - sd.cut_case_offsets[ 17 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 17 ][ 0 ], sd.cut_case_offsets[ 17 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 17 ][ 0 ], sd.cut_case_offsets[ 17 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 17 ][ 0 ], sd.cut_case_offsets[ 17 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 17 ][ 0 ], sd.cut_case_offsets[ 17 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 17 ].data(), index_best_sub_case, sd.cut_case_offsets[ 17 ][ 1 ] - sd.cut_case_offsets[ 17 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 17 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 18 ][ 1 ] - sd.cut_case_offsets[ 18 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 1, 1, 2,
            3, 4, 4, 5,
            3, 4, 1, 2,
            0, 1, 4, 5,
            0, 1, 4, 5,
            3, 4, 1, 2,
            3, 4, 4, 5,
            0, 1, 1, 2,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 18 ][ 1 ] - sd.cut_case_offsets[ 18 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 18 ][ 0 ], sd.cut_case_offsets[ 18 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 18 ][ 0 ], sd.cut_case_offsets[ 18 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 18 ][ 0 ], sd.cut_case_offsets[ 18 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 18 ][ 0 ], sd.cut_case_offsets[ 18 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 18 ].data(), index_best_sub_case, sd.cut_case_offsets[ 18 ][ 1 ] - sd.cut_case_offsets[ 18 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 18 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 19 ][ 1 ] - sd.cut_case_offsets[ 19 ][ 0 ] ) {
        static std::vector<BI> nv{
            5, 0, 4, 5,
            3, 4, 1, 2,
            3, 4, 4, 5,
            5, 0, 1, 2,
            5, 0, 1, 2,
            3, 4, 4, 5,
            3, 4, 1, 2,
            5, 0, 4, 5,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 19 ][ 1 ] - sd.cut_case_offsets[ 19 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 19 ][ 0 ], sd.cut_case_offsets[ 19 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 19 ][ 0 ], sd.cut_case_offsets[ 19 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 19 ][ 0 ], sd.cut_case_offsets[ 19 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 19 ][ 0 ], sd.cut_case_offsets[ 19 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 19 ].data(), index_best_sub_case, sd.cut_case_offsets[ 19 ][ 1 ] - sd.cut_case_offsets[ 19 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 19 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 20 ][ 1 ] - sd.cut_case_offsets[ 20 ][ 0 ] ) {
        static std::vector<BI> nv{
            1, 2, 2, 3,
            3, 4, 4, 5,
            3, 4, 2, 3,
            1, 2, 4, 5,
            1, 2, 4, 5,
            3, 4, 2, 3,
            3, 4, 4, 5,
            1, 2, 2, 3,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 20 ][ 1 ] - sd.cut_case_offsets[ 20 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 20 ][ 0 ], sd.cut_case_offsets[ 20 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 20 ][ 0 ], sd.cut_case_offsets[ 20 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 20 ][ 0 ], sd.cut_case_offsets[ 20 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 20 ][ 0 ], sd.cut_case_offsets[ 20 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 20 ].data(), index_best_sub_case, sd.cut_case_offsets[ 20 ][ 1 ] - sd.cut_case_offsets[ 20 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 20 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 21 ][ 1 ] - sd.cut_case_offsets[ 21 ][ 0 ] ) {
        static std::vector<BI> nv{
            5, 0, 4, 5,
            3, 4, 2, 3,
            1, 2, 0, 1,
            3, 4, 4, 5,
            5, 0, 2, 3,
            5, 0, 2, 3,
            3, 4, 4, 5,
            3, 4, 2, 3,
            5, 0, 4, 5,
            1, 2, 2, 3,
            3, 4, 4, 5,
            5, 0, 0, 1,
            5, 0, 4, 5,
            1, 2, 2, 3,
            3, 4, 0, 1,
            1, 2, 2, 3,
            3, 4, 0, 1,
            5, 0, 4, 5,
            5, 0, 0, 1,
            1, 2, 2, 3,
            3, 4, 4, 5,
            3, 4, 2, 3,
            1, 2, 4, 5,
            5, 0, 0, 1,
            5, 0, 4, 5,
            3, 4, 0, 1,
            1, 2, 2, 3,
            1, 2, 0, 1,
            5, 0, 4, 5,
            3, 4, 2, 3,
            5, 0, 0, 1,
            1, 2, 4, 5,
            1, 2, 4, 5,
            5, 0, 0, 1,
            5, 0, 4, 5,
            1, 2, 0, 1,
            3, 4, 4, 5,
            5, 0, 0, 1,
            1, 2, 2, 3,
            1, 2, 0, 1,
            3, 4, 4, 5,
            5, 0, 2, 3,
            3, 4, 4, 5,
            5, 0, 2, 3,
            1, 2, 0, 1,
            1, 2, 2, 3,
            3, 4, 4, 5,
            5, 0, 0, 1,
            5, 0, 0, 1,
            1, 2, 2, 3,
            3, 4, 4, 5,
            3, 4, 2, 3,
            5, 0, 0, 1,
            1, 2, 4, 5,
            5, 0, 0, 1,
            1, 2, 4, 5,
            3, 4, 2, 3,
            3, 4, 4, 5,
            5, 0, 0, 1,
            1, 2, 2, 3,
            1, 2, 0, 1,
            5, 0, 2, 3,
            3, 4, 4, 5,
            3, 4, 2, 3,
            1, 2, 0, 1,
            5, 0, 4, 5,
            1, 2, 2, 3,
            3, 4, 0, 1,
            3, 4, 0, 1,
            1, 2, 2, 3,
            1, 2, 0, 1,
            3, 4, 2, 3,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 21 ][ 1 ] - sd.cut_case_offsets[ 21 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 0, nn.data(), 0, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 1, nn.data(), 12, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 2, nn.data(), 20, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 3, nn.data(), 28, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 4, nn.data(), 36, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 5, nn.data(), 48, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 6, nn.data(), 60, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 7, nn.data(), 72, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 8, nn.data(), 84, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 9, nn.data(), 96, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 10, nn.data(), 108, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 11, nn.data(), 120, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 12, nn.data(), 128, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 13, nn.data(), 136, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 14, nn.data(), 144, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 15, nn.data(), 156, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 16, nn.data(), 168, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 17, nn.data(), 180, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 18, nn.data(), 192, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 19, nn.data(), 204, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 20, nn.data(), 216, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 21, nn.data(), 228, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 22, nn.data(), 240, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 23, nn.data(), 252, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 24, nn.data(), 264, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 25, nn.data(), 272, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 21 ][ 0 ], sd.cut_case_offsets[ 21 ][ 1 ], 26, nn.data(), 280, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 21 ].data(), index_best_sub_case, sd.cut_case_offsets[ 21 ][ 1 ] - sd.cut_case_offsets[ 21 ][ 0 ], 27, sd.cut_indices, sd.cut_case_offsets[ 21 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 22 ][ 1 ] - sd.cut_case_offsets[ 22 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 1, 2, 3,
            3, 4, 4, 5,
            3, 4, 2, 3,
            0, 1, 4, 5,
            0, 1, 4, 5,
            3, 4, 2, 3,
            3, 4, 4, 5,
            0, 1, 2, 3,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 22 ][ 1 ] - sd.cut_case_offsets[ 22 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 22 ][ 0 ], sd.cut_case_offsets[ 22 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 22 ][ 0 ], sd.cut_case_offsets[ 22 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 22 ][ 0 ], sd.cut_case_offsets[ 22 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 22 ][ 0 ], sd.cut_case_offsets[ 22 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 22 ].data(), index_best_sub_case, sd.cut_case_offsets[ 22 ][ 1 ] - sd.cut_case_offsets[ 22 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 22 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 23 ][ 1 ] - sd.cut_case_offsets[ 23 ][ 0 ] ) {
        static std::vector<BI> nv{
            5, 0, 4, 5,
            3, 4, 2, 3,
            3, 4, 4, 5,
            5, 0, 2, 3,
            5, 0, 2, 3,
            3, 4, 4, 5,
            3, 4, 2, 3,
            5, 0, 4, 5,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 23 ][ 1 ] - sd.cut_case_offsets[ 23 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 23 ][ 0 ], sd.cut_case_offsets[ 23 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 23 ][ 0 ], sd.cut_case_offsets[ 23 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 23 ][ 0 ], sd.cut_case_offsets[ 23 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 23 ][ 0 ], sd.cut_case_offsets[ 23 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 23 ].data(), index_best_sub_case, sd.cut_case_offsets[ 23 ][ 1 ] - sd.cut_case_offsets[ 23 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 23 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 25 ][ 1 ] - sd.cut_case_offsets[ 25 ][ 0 ] ) {
        static std::vector<BI> nv{
            5, 0, 4, 5,
            2, 3, 0, 1,
            2, 3, 4, 5,
            5, 0, 0, 1,
            5, 0, 0, 1,
            2, 3, 4, 5,
            2, 3, 0, 1,
            5, 0, 4, 5,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 25 ][ 1 ] - sd.cut_case_offsets[ 25 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 25 ][ 0 ], sd.cut_case_offsets[ 25 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 25 ][ 0 ], sd.cut_case_offsets[ 25 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 25 ][ 0 ], sd.cut_case_offsets[ 25 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 25 ][ 0 ], sd.cut_case_offsets[ 25 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 25 ].data(), index_best_sub_case, sd.cut_case_offsets[ 25 ][ 1 ] - sd.cut_case_offsets[ 25 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 25 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 26 ][ 1 ] - sd.cut_case_offsets[ 26 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 1, 1, 2,
            2, 3, 4, 5,
            2, 3, 1, 2,
            0, 1, 4, 5,
            0, 1, 4, 5,
            2, 3, 1, 2,
            2, 3, 4, 5,
            0, 1, 1, 2,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 26 ][ 1 ] - sd.cut_case_offsets[ 26 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 26 ][ 0 ], sd.cut_case_offsets[ 26 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 26 ][ 0 ], sd.cut_case_offsets[ 26 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 26 ][ 0 ], sd.cut_case_offsets[ 26 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 26 ][ 0 ], sd.cut_case_offsets[ 26 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 26 ].data(), index_best_sub_case, sd.cut_case_offsets[ 26 ][ 1 ] - sd.cut_case_offsets[ 26 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 26 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 27 ][ 1 ] - sd.cut_case_offsets[ 27 ][ 0 ] ) {
        static std::vector<BI> nv{
            5, 0, 4, 5,
            2, 3, 1, 2,
            2, 3, 4, 5,
            5, 0, 1, 2,
            5, 0, 1, 2,
            2, 3, 4, 5,
            2, 3, 1, 2,
            5, 0, 4, 5,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 27 ][ 1 ] - sd.cut_case_offsets[ 27 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 27 ][ 0 ], sd.cut_case_offsets[ 27 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 27 ][ 0 ], sd.cut_case_offsets[ 27 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 27 ][ 0 ], sd.cut_case_offsets[ 27 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 27 ][ 0 ], sd.cut_case_offsets[ 27 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 27 ].data(), index_best_sub_case, sd.cut_case_offsets[ 27 ][ 1 ] - sd.cut_case_offsets[ 27 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 27 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 29 ][ 1 ] - sd.cut_case_offsets[ 29 ][ 0 ] ) {
        static std::vector<BI> nv{
            5, 0, 4, 5,
            1, 2, 0, 1,
            1, 2, 4, 5,
            5, 0, 0, 1,
            5, 0, 0, 1,
            1, 2, 4, 5,
            1, 2, 0, 1,
            5, 0, 4, 5,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 29 ][ 1 ] - sd.cut_case_offsets[ 29 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 29 ][ 0 ], sd.cut_case_offsets[ 29 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 29 ][ 0 ], sd.cut_case_offsets[ 29 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 29 ][ 0 ], sd.cut_case_offsets[ 29 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 29 ][ 0 ], sd.cut_case_offsets[ 29 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 29 ].data(), index_best_sub_case, sd.cut_case_offsets[ 29 ][ 1 ] - sd.cut_case_offsets[ 29 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 29 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 34 ][ 1 ] - sd.cut_case_offsets[ 34 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 1, 1, 2,
            4, 5, 5, 0,
            4, 5, 1, 2,
            0, 1, 5, 0,
            0, 1, 5, 0,
            4, 5, 1, 2,
            4, 5, 5, 0,
            0, 1, 1, 2,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 34 ][ 1 ] - sd.cut_case_offsets[ 34 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 34 ][ 0 ], sd.cut_case_offsets[ 34 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 34 ][ 0 ], sd.cut_case_offsets[ 34 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 34 ][ 0 ], sd.cut_case_offsets[ 34 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 34 ][ 0 ], sd.cut_case_offsets[ 34 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 34 ].data(), index_best_sub_case, sd.cut_case_offsets[ 34 ][ 1 ] - sd.cut_case_offsets[ 34 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 34 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 36 ][ 1 ] - sd.cut_case_offsets[ 36 ][ 0 ] ) {
        static std::vector<BI> nv{
            1, 2, 2, 3,
            4, 5, 5, 0,
            4, 5, 2, 3,
            1, 2, 5, 0,
            1, 2, 5, 0,
            4, 5, 2, 3,
            4, 5, 5, 0,
            1, 2, 2, 3,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 36 ][ 1 ] - sd.cut_case_offsets[ 36 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 36 ][ 0 ], sd.cut_case_offsets[ 36 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 36 ][ 0 ], sd.cut_case_offsets[ 36 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 36 ][ 0 ], sd.cut_case_offsets[ 36 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 36 ][ 0 ], sd.cut_case_offsets[ 36 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 36 ].data(), index_best_sub_case, sd.cut_case_offsets[ 36 ][ 1 ] - sd.cut_case_offsets[ 36 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 36 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 37 ][ 1 ] - sd.cut_case_offsets[ 37 ][ 0 ] ) {
        static std::vector<BI> nv{
            4, 5, 2, 3,
            1, 2, 0, 1,
            1, 2, 2, 3,
            4, 5, 0, 1,
            4, 5, 0, 1,
            1, 2, 2, 3,
            1, 2, 0, 1,
            4, 5, 2, 3,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 37 ][ 1 ] - sd.cut_case_offsets[ 37 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 37 ][ 0 ], sd.cut_case_offsets[ 37 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 37 ][ 0 ], sd.cut_case_offsets[ 37 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 37 ][ 0 ], sd.cut_case_offsets[ 37 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 37 ][ 0 ], sd.cut_case_offsets[ 37 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 37 ].data(), index_best_sub_case, sd.cut_case_offsets[ 37 ][ 1 ] - sd.cut_case_offsets[ 37 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 37 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 38 ][ 1 ] - sd.cut_case_offsets[ 38 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 1, 2, 3,
            4, 5, 5, 0,
            4, 5, 2, 3,
            0, 1, 5, 0,
            0, 1, 5, 0,
            4, 5, 2, 3,
            4, 5, 5, 0,
            0, 1, 2, 3,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 38 ][ 1 ] - sd.cut_case_offsets[ 38 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 38 ][ 0 ], sd.cut_case_offsets[ 38 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 38 ][ 0 ], sd.cut_case_offsets[ 38 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 38 ][ 0 ], sd.cut_case_offsets[ 38 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 38 ][ 0 ], sd.cut_case_offsets[ 38 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 38 ].data(), index_best_sub_case, sd.cut_case_offsets[ 38 ][ 1 ] - sd.cut_case_offsets[ 38 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 38 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 40 ][ 1 ] - sd.cut_case_offsets[ 40 ][ 0 ] ) {
        static std::vector<BI> nv{
            2, 3, 3, 4,
            4, 5, 5, 0,
            4, 5, 3, 4,
            2, 3, 5, 0,
            2, 3, 5, 0,
            4, 5, 3, 4,
            4, 5, 5, 0,
            2, 3, 3, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 40 ][ 1 ] - sd.cut_case_offsets[ 40 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 40 ][ 0 ], sd.cut_case_offsets[ 40 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 40 ][ 0 ], sd.cut_case_offsets[ 40 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 40 ][ 0 ], sd.cut_case_offsets[ 40 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 40 ][ 0 ], sd.cut_case_offsets[ 40 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 40 ].data(), index_best_sub_case, sd.cut_case_offsets[ 40 ][ 1 ] - sd.cut_case_offsets[ 40 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 40 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 41 ][ 1 ] - sd.cut_case_offsets[ 41 ][ 0 ] ) {
        static std::vector<BI> nv{
            4, 5, 3, 4,
            2, 3, 0, 1,
            2, 3, 3, 4,
            4, 5, 0, 1,
            4, 5, 0, 1,
            2, 3, 3, 4,
            2, 3, 0, 1,
            4, 5, 3, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 41 ][ 1 ] - sd.cut_case_offsets[ 41 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 41 ][ 0 ], sd.cut_case_offsets[ 41 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 41 ][ 0 ], sd.cut_case_offsets[ 41 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 41 ][ 0 ], sd.cut_case_offsets[ 41 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 41 ][ 0 ], sd.cut_case_offsets[ 41 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 41 ].data(), index_best_sub_case, sd.cut_case_offsets[ 41 ][ 1 ] - sd.cut_case_offsets[ 41 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 41 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 42 ][ 1 ] - sd.cut_case_offsets[ 42 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 1, 1, 2,
            2, 3, 3, 4,
            4, 5, 5, 0,
            4, 5, 3, 4,
            0, 1, 1, 2,
            2, 3, 5, 0,
            0, 1, 1, 2,
            2, 3, 5, 0,
            4, 5, 3, 4,
            4, 5, 5, 0,
            0, 1, 1, 2,
            2, 3, 3, 4,
            2, 3, 1, 2,
            0, 1, 3, 4,
            4, 5, 5, 0,
            4, 5, 3, 4,
            2, 3, 1, 2,
            0, 1, 5, 0,
            2, 3, 3, 4,
            4, 5, 1, 2,
            4, 5, 1, 2,
            2, 3, 3, 4,
            2, 3, 1, 2,
            4, 5, 3, 4,
            0, 1, 5, 0,
            4, 5, 3, 4,
            2, 3, 1, 2,
            4, 5, 5, 0,
            0, 1, 3, 4,
            0, 1, 3, 4,
            4, 5, 5, 0,
            4, 5, 3, 4,
            0, 1, 5, 0,
            2, 3, 3, 4,
            4, 5, 5, 0,
            0, 1, 1, 2,
            0, 1, 5, 0,
            2, 3, 3, 4,
            4, 5, 1, 2,
            2, 3, 3, 4,
            4, 5, 1, 2,
            0, 1, 5, 0,
            0, 1, 1, 2,
            2, 3, 3, 4,
            4, 5, 5, 0,
            4, 5, 3, 4,
            2, 3, 5, 0,
            0, 1, 1, 2,
            0, 1, 5, 0,
            4, 5, 1, 2,
            2, 3, 3, 4,
            2, 3, 1, 2,
            0, 1, 5, 0,
            4, 5, 3, 4,
            0, 1, 1, 2,
            2, 3, 5, 0,
            2, 3, 5, 0,
            0, 1, 1, 2,
            0, 1, 5, 0,
            2, 3, 1, 2,
            4, 5, 5, 0,
            0, 1, 1, 2,
            2, 3, 3, 4,
            2, 3, 1, 2,
            4, 5, 5, 0,
            0, 1, 3, 4,
            4, 5, 5, 0,
            0, 1, 3, 4,
            2, 3, 1, 2,
            2, 3, 3, 4,
            4, 5, 5, 0,
            0, 1, 1, 2,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 42 ][ 1 ] - sd.cut_case_offsets[ 42 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 0, nn.data(), 0, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 1, nn.data(), 12, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 2, nn.data(), 24, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 3, nn.data(), 36, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 4, nn.data(), 48, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 5, nn.data(), 60, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 6, nn.data(), 72, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 7, nn.data(), 80, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 8, nn.data(), 88, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 9, nn.data(), 96, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 10, nn.data(), 108, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 11, nn.data(), 116, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 12, nn.data(), 124, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 13, nn.data(), 132, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 14, nn.data(), 144, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 15, nn.data(), 156, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 16, nn.data(), 168, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 17, nn.data(), 180, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 18, nn.data(), 192, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 19, nn.data(), 204, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 20, nn.data(), 216, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 21, nn.data(), 224, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 22, nn.data(), 232, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 23, nn.data(), 240, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 24, nn.data(), 252, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 25, nn.data(), 264, 3, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 42 ][ 0 ], sd.cut_case_offsets[ 42 ][ 1 ], 26, nn.data(), 276, 3, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 42 ].data(), index_best_sub_case, sd.cut_case_offsets[ 42 ][ 1 ] - sd.cut_case_offsets[ 42 ][ 0 ], 27, sd.cut_indices, sd.cut_case_offsets[ 42 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 43 ][ 1 ] - sd.cut_case_offsets[ 43 ][ 0 ] ) {
        static std::vector<BI> nv{
            4, 5, 3, 4,
            2, 3, 1, 2,
            2, 3, 3, 4,
            4, 5, 1, 2,
            4, 5, 1, 2,
            2, 3, 3, 4,
            2, 3, 1, 2,
            4, 5, 3, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 43 ][ 1 ] - sd.cut_case_offsets[ 43 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 43 ][ 0 ], sd.cut_case_offsets[ 43 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 43 ][ 0 ], sd.cut_case_offsets[ 43 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 43 ][ 0 ], sd.cut_case_offsets[ 43 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 43 ][ 0 ], sd.cut_case_offsets[ 43 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 43 ].data(), index_best_sub_case, sd.cut_case_offsets[ 43 ][ 1 ] - sd.cut_case_offsets[ 43 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 43 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 44 ][ 1 ] - sd.cut_case_offsets[ 44 ][ 0 ] ) {
        static std::vector<BI> nv{
            1, 2, 3, 4,
            4, 5, 5, 0,
            4, 5, 3, 4,
            1, 2, 5, 0,
            1, 2, 5, 0,
            4, 5, 3, 4,
            4, 5, 5, 0,
            1, 2, 3, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 44 ][ 1 ] - sd.cut_case_offsets[ 44 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 44 ][ 0 ], sd.cut_case_offsets[ 44 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 44 ][ 0 ], sd.cut_case_offsets[ 44 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 44 ][ 0 ], sd.cut_case_offsets[ 44 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 44 ][ 0 ], sd.cut_case_offsets[ 44 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 44 ].data(), index_best_sub_case, sd.cut_case_offsets[ 44 ][ 1 ] - sd.cut_case_offsets[ 44 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 44 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 45 ][ 1 ] - sd.cut_case_offsets[ 45 ][ 0 ] ) {
        static std::vector<BI> nv{
            4, 5, 3, 4,
            1, 2, 0, 1,
            1, 2, 3, 4,
            4, 5, 0, 1,
            4, 5, 0, 1,
            1, 2, 3, 4,
            1, 2, 0, 1,
            4, 5, 3, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 45 ][ 1 ] - sd.cut_case_offsets[ 45 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 45 ][ 0 ], sd.cut_case_offsets[ 45 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 45 ][ 0 ], sd.cut_case_offsets[ 45 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 45 ][ 0 ], sd.cut_case_offsets[ 45 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 45 ][ 0 ], sd.cut_case_offsets[ 45 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 45 ].data(), index_best_sub_case, sd.cut_case_offsets[ 45 ][ 1 ] - sd.cut_case_offsets[ 45 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 45 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 46 ][ 1 ] - sd.cut_case_offsets[ 46 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 1, 3, 4,
            4, 5, 5, 0,
            4, 5, 3, 4,
            0, 1, 5, 0,
            0, 1, 5, 0,
            4, 5, 3, 4,
            4, 5, 5, 0,
            0, 1, 3, 4,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 46 ][ 1 ] - sd.cut_case_offsets[ 46 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 46 ][ 0 ], sd.cut_case_offsets[ 46 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 46 ][ 0 ], sd.cut_case_offsets[ 46 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 46 ][ 0 ], sd.cut_case_offsets[ 46 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 46 ][ 0 ], sd.cut_case_offsets[ 46 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 46 ].data(), index_best_sub_case, sd.cut_case_offsets[ 46 ][ 1 ] - sd.cut_case_offsets[ 46 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 46 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 50 ][ 1 ] - sd.cut_case_offsets[ 50 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 1, 1, 2,
            3, 4, 5, 0,
            3, 4, 1, 2,
            0, 1, 5, 0,
            0, 1, 5, 0,
            3, 4, 1, 2,
            3, 4, 5, 0,
            0, 1, 1, 2,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 50 ][ 1 ] - sd.cut_case_offsets[ 50 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 50 ][ 0 ], sd.cut_case_offsets[ 50 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 50 ][ 0 ], sd.cut_case_offsets[ 50 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 50 ][ 0 ], sd.cut_case_offsets[ 50 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 50 ][ 0 ], sd.cut_case_offsets[ 50 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 50 ].data(), index_best_sub_case, sd.cut_case_offsets[ 50 ][ 1 ] - sd.cut_case_offsets[ 50 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 50 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 52 ][ 1 ] - sd.cut_case_offsets[ 52 ][ 0 ] ) {
        static std::vector<BI> nv{
            1, 2, 2, 3,
            3, 4, 5, 0,
            3, 4, 2, 3,
            1, 2, 5, 0,
            1, 2, 5, 0,
            3, 4, 2, 3,
            3, 4, 5, 0,
            1, 2, 2, 3,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 52 ][ 1 ] - sd.cut_case_offsets[ 52 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 52 ][ 0 ], sd.cut_case_offsets[ 52 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 52 ][ 0 ], sd.cut_case_offsets[ 52 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 52 ][ 0 ], sd.cut_case_offsets[ 52 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 52 ][ 0 ], sd.cut_case_offsets[ 52 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 52 ].data(), index_best_sub_case, sd.cut_case_offsets[ 52 ][ 1 ] - sd.cut_case_offsets[ 52 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 52 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 53 ][ 1 ] - sd.cut_case_offsets[ 53 ][ 0 ] ) {
        static std::vector<BI> nv{
            3, 4, 2, 3,
            1, 2, 0, 1,
            1, 2, 2, 3,
            3, 4, 0, 1,
            3, 4, 0, 1,
            1, 2, 2, 3,
            1, 2, 0, 1,
            3, 4, 2, 3,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 53 ][ 1 ] - sd.cut_case_offsets[ 53 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 53 ][ 0 ], sd.cut_case_offsets[ 53 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 53 ][ 0 ], sd.cut_case_offsets[ 53 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 53 ][ 0 ], sd.cut_case_offsets[ 53 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 53 ][ 0 ], sd.cut_case_offsets[ 53 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 53 ].data(), index_best_sub_case, sd.cut_case_offsets[ 53 ][ 1 ] - sd.cut_case_offsets[ 53 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 53 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 54 ][ 1 ] - sd.cut_case_offsets[ 54 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 1, 2, 3,
            3, 4, 5, 0,
            3, 4, 2, 3,
            0, 1, 5, 0,
            0, 1, 5, 0,
            3, 4, 2, 3,
            3, 4, 5, 0,
            0, 1, 2, 3,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 54 ][ 1 ] - sd.cut_case_offsets[ 54 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 54 ][ 0 ], sd.cut_case_offsets[ 54 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 54 ][ 0 ], sd.cut_case_offsets[ 54 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 54 ][ 0 ], sd.cut_case_offsets[ 54 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 54 ][ 0 ], sd.cut_case_offsets[ 54 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 54 ].data(), index_best_sub_case, sd.cut_case_offsets[ 54 ][ 1 ] - sd.cut_case_offsets[ 54 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 54 ][ 0 ] );
    }

    if ( sd.cut_case_offsets[ 58 ][ 1 ] - sd.cut_case_offsets[ 58 ][ 0 ] ) {
        static std::vector<BI> nv{
            0, 1, 1, 2,
            2, 3, 5, 0,
            2, 3, 1, 2,
            0, 1, 5, 0,
            0, 1, 5, 0,
            2, 3, 1, 2,
            2, 3, 5, 0,
            0, 1, 1, 2,
        };

        VecTI nn{ ks, nv };
        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 58 ][ 1 ] - sd.cut_case_offsets[ 58 ][ 0 ] );

        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 58 ][ 0 ], sd.cut_case_offsets[ 58 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 58 ][ 0 ], sd.cut_case_offsets[ 58 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 58 ][ 0 ], sd.cut_case_offsets[ 58 ][ 1 ], 2, nn.data(), 16, 2, N<2>() );
        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 58 ][ 0 ], sd.cut_case_offsets[ 58 ][ 1 ], 3, nn.data(), 24, 2, N<2>() );
        ks->sort_TI_in_range( sd.cut_case_offsets[ 58 ].data(), index_best_sub_case, sd.cut_case_offsets[ 58 ][ 1 ] - sd.cut_case_offsets[ 58 ][ 0 ], 4, sd.cut_indices, sd.cut_case_offsets[ 58 ][ 0 ] );
    }

    fc( s3(),
        ( sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 5 ][ 4 ] - sd.cut_case_offsets[ 5 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 10 ][ 2 ] - sd.cut_case_offsets[ 10 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 10 ][ 3 ] - sd.cut_case_offsets[ 10 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 11 ][ 1 ] - sd.cut_case_offsets[ 11 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 11 ][ 4 ] - sd.cut_case_offsets[ 11 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 13 ][ 1 ] - sd.cut_case_offsets[ 13 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 13 ][ 4 ] - sd.cut_case_offsets[ 13 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 17 ][ 1 ] - sd.cut_case_offsets[ 17 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 17 ][ 4 ] - sd.cut_case_offsets[ 17 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 19 ][ 1 ] - sd.cut_case_offsets[ 19 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 19 ][ 4 ] - sd.cut_case_offsets[ 19 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 20 ][ 2 ] - sd.cut_case_offsets[ 20 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 20 ][ 3 ] - sd.cut_case_offsets[ 20 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 1 ] - sd.cut_case_offsets[ 21 ][ 0 ] ) * 3 +
        ( sd.cut_case_offsets[ 21 ][ 4 ] - sd.cut_case_offsets[ 21 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 21 ][ 6 ] - sd.cut_case_offsets[ 21 ][ 5 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 7 ] - sd.cut_case_offsets[ 21 ][ 6 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 9 ] - sd.cut_case_offsets[ 21 ][ 8 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 10 ] - sd.cut_case_offsets[ 21 ][ 9 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 11 ] - sd.cut_case_offsets[ 21 ][ 10 ] ) * 3 +
        ( sd.cut_case_offsets[ 21 ][ 14 ] - sd.cut_case_offsets[ 21 ][ 13 ] ) * 2 +
        ( sd.cut_case_offsets[ 21 ][ 16 ] - sd.cut_case_offsets[ 21 ][ 15 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 17 ] - sd.cut_case_offsets[ 21 ][ 16 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 20 ] - sd.cut_case_offsets[ 21 ][ 19 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 21 ] - sd.cut_case_offsets[ 21 ][ 20 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 23 ] - sd.cut_case_offsets[ 21 ][ 22 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 24 ] - sd.cut_case_offsets[ 21 ][ 23 ] ) * 3 +
        ( sd.cut_case_offsets[ 21 ][ 27 ] - sd.cut_case_offsets[ 21 ][ 26 ] ) * 2 +
        ( sd.cut_case_offsets[ 22 ][ 2 ] - sd.cut_case_offsets[ 22 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 22 ][ 3 ] - sd.cut_case_offsets[ 22 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 23 ][ 1 ] - sd.cut_case_offsets[ 23 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 23 ][ 4 ] - sd.cut_case_offsets[ 23 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 25 ][ 1 ] - sd.cut_case_offsets[ 25 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 25 ][ 4 ] - sd.cut_case_offsets[ 25 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 26 ][ 2 ] - sd.cut_case_offsets[ 26 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 26 ][ 3 ] - sd.cut_case_offsets[ 26 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 27 ][ 1 ] - sd.cut_case_offsets[ 27 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 27 ][ 4 ] - sd.cut_case_offsets[ 27 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 29 ][ 1 ] - sd.cut_case_offsets[ 29 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 29 ][ 4 ] - sd.cut_case_offsets[ 29 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 31 ][ 1 ] - sd.cut_case_offsets[ 31 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 34 ][ 2 ] - sd.cut_case_offsets[ 34 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 34 ][ 3 ] - sd.cut_case_offsets[ 34 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 37 ][ 1 ] - sd.cut_case_offsets[ 37 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 37 ][ 4 ] - sd.cut_case_offsets[ 37 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 38 ][ 2 ] - sd.cut_case_offsets[ 38 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 38 ][ 3 ] - sd.cut_case_offsets[ 38 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 40 ][ 2 ] - sd.cut_case_offsets[ 40 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 40 ][ 3 ] - sd.cut_case_offsets[ 40 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 41 ][ 1 ] - sd.cut_case_offsets[ 41 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 41 ][ 4 ] - sd.cut_case_offsets[ 41 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 2 ] - sd.cut_case_offsets[ 42 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 3 ] - sd.cut_case_offsets[ 42 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 5 ] - sd.cut_case_offsets[ 42 ][ 4 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 6 ] - sd.cut_case_offsets[ 42 ][ 5 ] ) * 3 +
        ( sd.cut_case_offsets[ 42 ][ 9 ] - sd.cut_case_offsets[ 42 ][ 8 ] ) * 2 +
        ( sd.cut_case_offsets[ 42 ][ 10 ] - sd.cut_case_offsets[ 42 ][ 9 ] ) * 3 +
        ( sd.cut_case_offsets[ 42 ][ 13 ] - sd.cut_case_offsets[ 42 ][ 12 ] ) * 2 +
        ( sd.cut_case_offsets[ 42 ][ 15 ] - sd.cut_case_offsets[ 42 ][ 14 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 16 ] - sd.cut_case_offsets[ 42 ][ 15 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 18 ] - sd.cut_case_offsets[ 42 ][ 17 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 19 ] - sd.cut_case_offsets[ 42 ][ 18 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 20 ] - sd.cut_case_offsets[ 42 ][ 19 ] ) * 3 +
        ( sd.cut_case_offsets[ 42 ][ 23 ] - sd.cut_case_offsets[ 42 ][ 22 ] ) * 2 +
        ( sd.cut_case_offsets[ 42 ][ 25 ] - sd.cut_case_offsets[ 42 ][ 24 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 26 ] - sd.cut_case_offsets[ 42 ][ 25 ] ) * 1 +
        ( sd.cut_case_offsets[ 43 ][ 1 ] - sd.cut_case_offsets[ 43 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 43 ][ 4 ] - sd.cut_case_offsets[ 43 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 44 ][ 2 ] - sd.cut_case_offsets[ 44 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 44 ][ 3 ] - sd.cut_case_offsets[ 44 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 45 ][ 1 ] - sd.cut_case_offsets[ 45 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 45 ][ 4 ] - sd.cut_case_offsets[ 45 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 46 ][ 2 ] - sd.cut_case_offsets[ 46 ][ 1 ] ) * 2 +
        ( sd.cut_case_offsets[ 46 ][ 3 ] - sd.cut_case_offsets[ 46 ][ 2 ] ) * 2 +
        ( sd.cut_case_offsets[ 47 ][ 1 ] - sd.cut_case_offsets[ 47 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 50 ][ 2 ] - sd.cut_case_offsets[ 50 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 50 ][ 3 ] - sd.cut_case_offsets[ 50 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 52 ][ 2 ] - sd.cut_case_offsets[ 52 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 52 ][ 3 ] - sd.cut_case_offsets[ 52 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 53 ][ 1 ] - sd.cut_case_offsets[ 53 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 53 ][ 4 ] - sd.cut_case_offsets[ 53 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 54 ][ 2 ] - sd.cut_case_offsets[ 54 ][ 1 ] ) * 2 +
        ( sd.cut_case_offsets[ 54 ][ 3 ] - sd.cut_case_offsets[ 54 ][ 2 ] ) * 2 +
        ( sd.cut_case_offsets[ 55 ][ 1 ] - sd.cut_case_offsets[ 55 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 58 ][ 2 ] - sd.cut_case_offsets[ 58 ][ 1 ] ) * 2 +
        ( sd.cut_case_offsets[ 58 ][ 3 ] - sd.cut_case_offsets[ 58 ][ 2 ] ) * 2 +
        ( sd.cut_case_offsets[ 59 ][ 1 ] - sd.cut_case_offsets[ 59 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 61 ][ 1 ] - sd.cut_case_offsets[ 61 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 62 ][ 1 ] - sd.cut_case_offsets[ 62 ][ 0 ] ) * 1
    );

    fc( s4(),
        ( sd.cut_case_offsets[ 1 ][ 1 ] - sd.cut_case_offsets[ 1 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 2 ][ 1 ] - sd.cut_case_offsets[ 2 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 4 ][ 1 ] - sd.cut_case_offsets[ 4 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 8 ][ 1 ] - sd.cut_case_offsets[ 8 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 9 ][ 1 ] - sd.cut_case_offsets[ 9 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 9 ][ 4 ] - sd.cut_case_offsets[ 9 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 11 ][ 1 ] - sd.cut_case_offsets[ 11 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 11 ][ 2 ] - sd.cut_case_offsets[ 11 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 11 ][ 3 ] - sd.cut_case_offsets[ 11 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 11 ][ 4 ] - sd.cut_case_offsets[ 11 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 13 ][ 1 ] - sd.cut_case_offsets[ 13 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 13 ][ 2 ] - sd.cut_case_offsets[ 13 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 13 ][ 3 ] - sd.cut_case_offsets[ 13 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 13 ][ 4 ] - sd.cut_case_offsets[ 13 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 15 ][ 1 ] - sd.cut_case_offsets[ 15 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 16 ][ 1 ] - sd.cut_case_offsets[ 16 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 18 ][ 2 ] - sd.cut_case_offsets[ 18 ][ 1 ] ) * 2 +
        ( sd.cut_case_offsets[ 18 ][ 3 ] - sd.cut_case_offsets[ 18 ][ 2 ] ) * 2 +
        ( sd.cut_case_offsets[ 19 ][ 1 ] - sd.cut_case_offsets[ 19 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 19 ][ 2 ] - sd.cut_case_offsets[ 19 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 19 ][ 3 ] - sd.cut_case_offsets[ 19 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 19 ][ 4 ] - sd.cut_case_offsets[ 19 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 22 ][ 1 ] - sd.cut_case_offsets[ 22 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 22 ][ 2 ] - sd.cut_case_offsets[ 22 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 22 ][ 3 ] - sd.cut_case_offsets[ 22 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 22 ][ 4 ] - sd.cut_case_offsets[ 22 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 25 ][ 1 ] - sd.cut_case_offsets[ 25 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 25 ][ 2 ] - sd.cut_case_offsets[ 25 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 25 ][ 3 ] - sd.cut_case_offsets[ 25 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 25 ][ 4 ] - sd.cut_case_offsets[ 25 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 26 ][ 1 ] - sd.cut_case_offsets[ 26 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 26 ][ 2 ] - sd.cut_case_offsets[ 26 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 26 ][ 3 ] - sd.cut_case_offsets[ 26 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 26 ][ 4 ] - sd.cut_case_offsets[ 26 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 30 ][ 1 ] - sd.cut_case_offsets[ 30 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 32 ][ 1 ] - sd.cut_case_offsets[ 32 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 36 ][ 2 ] - sd.cut_case_offsets[ 36 ][ 1 ] ) * 2 +
        ( sd.cut_case_offsets[ 36 ][ 3 ] - sd.cut_case_offsets[ 36 ][ 2 ] ) * 2 +
        ( sd.cut_case_offsets[ 37 ][ 1 ] - sd.cut_case_offsets[ 37 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 37 ][ 2 ] - sd.cut_case_offsets[ 37 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 37 ][ 3 ] - sd.cut_case_offsets[ 37 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 37 ][ 4 ] - sd.cut_case_offsets[ 37 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 38 ][ 1 ] - sd.cut_case_offsets[ 38 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 38 ][ 2 ] - sd.cut_case_offsets[ 38 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 38 ][ 3 ] - sd.cut_case_offsets[ 38 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 38 ][ 4 ] - sd.cut_case_offsets[ 38 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 39 ][ 1 ] - sd.cut_case_offsets[ 39 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 41 ][ 1 ] - sd.cut_case_offsets[ 41 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 41 ][ 2 ] - sd.cut_case_offsets[ 41 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 41 ][ 3 ] - sd.cut_case_offsets[ 41 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 41 ][ 4 ] - sd.cut_case_offsets[ 41 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 44 ][ 1 ] - sd.cut_case_offsets[ 44 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 44 ][ 2 ] - sd.cut_case_offsets[ 44 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 44 ][ 3 ] - sd.cut_case_offsets[ 44 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 44 ][ 4 ] - sd.cut_case_offsets[ 44 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 50 ][ 1 ] - sd.cut_case_offsets[ 50 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 50 ][ 2 ] - sd.cut_case_offsets[ 50 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 50 ][ 3 ] - sd.cut_case_offsets[ 50 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 50 ][ 4 ] - sd.cut_case_offsets[ 50 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 51 ][ 1 ] - sd.cut_case_offsets[ 51 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 52 ][ 1 ] - sd.cut_case_offsets[ 52 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 52 ][ 2 ] - sd.cut_case_offsets[ 52 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 52 ][ 3 ] - sd.cut_case_offsets[ 52 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 52 ][ 4 ] - sd.cut_case_offsets[ 52 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 57 ][ 1 ] - sd.cut_case_offsets[ 57 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 60 ][ 1 ] - sd.cut_case_offsets[ 60 ][ 0 ] ) * 1
    );

    fc( s5(),
        ( sd.cut_case_offsets[ 1 ][ 1 ] - sd.cut_case_offsets[ 1 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 2 ][ 1 ] - sd.cut_case_offsets[ 2 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 4 ][ 1 ] - sd.cut_case_offsets[ 4 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 5 ][ 2 ] - sd.cut_case_offsets[ 5 ][ 1 ] ) * 2 +
        ( sd.cut_case_offsets[ 5 ][ 3 ] - sd.cut_case_offsets[ 5 ][ 2 ] ) * 2 +
        ( sd.cut_case_offsets[ 5 ][ 4 ] - sd.cut_case_offsets[ 5 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 7 ][ 1 ] - sd.cut_case_offsets[ 7 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 8 ][ 1 ] - sd.cut_case_offsets[ 8 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 9 ][ 2 ] - sd.cut_case_offsets[ 9 ][ 1 ] ) * 2 +
        ( sd.cut_case_offsets[ 9 ][ 3 ] - sd.cut_case_offsets[ 9 ][ 2 ] ) * 2 +
        ( sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 10 ][ 2 ] - sd.cut_case_offsets[ 10 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 10 ][ 3 ] - sd.cut_case_offsets[ 10 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 10 ][ 4 ] - sd.cut_case_offsets[ 10 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 11 ][ 2 ] - sd.cut_case_offsets[ 11 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 11 ][ 3 ] - sd.cut_case_offsets[ 11 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 13 ][ 2 ] - sd.cut_case_offsets[ 13 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 13 ][ 3 ] - sd.cut_case_offsets[ 13 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 14 ][ 1 ] - sd.cut_case_offsets[ 14 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 16 ][ 1 ] - sd.cut_case_offsets[ 16 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 17 ][ 1 ] - sd.cut_case_offsets[ 17 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 17 ][ 2 ] - sd.cut_case_offsets[ 17 ][ 1 ] ) * 2 +
        ( sd.cut_case_offsets[ 17 ][ 3 ] - sd.cut_case_offsets[ 17 ][ 2 ] ) * 2 +
        ( sd.cut_case_offsets[ 17 ][ 4 ] - sd.cut_case_offsets[ 17 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 18 ][ 1 ] - sd.cut_case_offsets[ 18 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 18 ][ 4 ] - sd.cut_case_offsets[ 18 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 19 ][ 2 ] - sd.cut_case_offsets[ 19 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 19 ][ 3 ] - sd.cut_case_offsets[ 19 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 20 ][ 1 ] - sd.cut_case_offsets[ 20 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 20 ][ 2 ] - sd.cut_case_offsets[ 20 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 20 ][ 3 ] - sd.cut_case_offsets[ 20 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 20 ][ 4 ] - sd.cut_case_offsets[ 20 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 21 ][ 5 ] - sd.cut_case_offsets[ 21 ][ 4 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 8 ] - sd.cut_case_offsets[ 21 ][ 7 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 15 ] - sd.cut_case_offsets[ 21 ][ 14 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 18 ] - sd.cut_case_offsets[ 21 ][ 17 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 19 ] - sd.cut_case_offsets[ 21 ][ 18 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 22 ] - sd.cut_case_offsets[ 21 ][ 21 ] ) * 1 +
        ( sd.cut_case_offsets[ 22 ][ 1 ] - sd.cut_case_offsets[ 22 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 22 ][ 4 ] - sd.cut_case_offsets[ 22 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 25 ][ 2 ] - sd.cut_case_offsets[ 25 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 25 ][ 3 ] - sd.cut_case_offsets[ 25 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 26 ][ 1 ] - sd.cut_case_offsets[ 26 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 26 ][ 4 ] - sd.cut_case_offsets[ 26 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 28 ][ 1 ] - sd.cut_case_offsets[ 28 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 32 ][ 1 ] - sd.cut_case_offsets[ 32 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 34 ][ 1 ] - sd.cut_case_offsets[ 34 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 34 ][ 2 ] - sd.cut_case_offsets[ 34 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 34 ][ 3 ] - sd.cut_case_offsets[ 34 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 34 ][ 4 ] - sd.cut_case_offsets[ 34 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 35 ][ 1 ] - sd.cut_case_offsets[ 35 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 36 ][ 1 ] - sd.cut_case_offsets[ 36 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 36 ][ 4 ] - sd.cut_case_offsets[ 36 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 37 ][ 2 ] - sd.cut_case_offsets[ 37 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 37 ][ 3 ] - sd.cut_case_offsets[ 37 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 38 ][ 1 ] - sd.cut_case_offsets[ 38 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 38 ][ 4 ] - sd.cut_case_offsets[ 38 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 40 ][ 1 ] - sd.cut_case_offsets[ 40 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 40 ][ 2 ] - sd.cut_case_offsets[ 40 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 40 ][ 3 ] - sd.cut_case_offsets[ 40 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 40 ][ 4 ] - sd.cut_case_offsets[ 40 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 41 ][ 2 ] - sd.cut_case_offsets[ 41 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 41 ][ 3 ] - sd.cut_case_offsets[ 41 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 1 ] - sd.cut_case_offsets[ 42 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 4 ] - sd.cut_case_offsets[ 42 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 14 ] - sd.cut_case_offsets[ 42 ][ 13 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 17 ] - sd.cut_case_offsets[ 42 ][ 16 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 24 ] - sd.cut_case_offsets[ 42 ][ 23 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 27 ] - sd.cut_case_offsets[ 42 ][ 26 ] ) * 1 +
        ( sd.cut_case_offsets[ 44 ][ 1 ] - sd.cut_case_offsets[ 44 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 44 ][ 4 ] - sd.cut_case_offsets[ 44 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 49 ][ 1 ] - sd.cut_case_offsets[ 49 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 50 ][ 1 ] - sd.cut_case_offsets[ 50 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 50 ][ 4 ] - sd.cut_case_offsets[ 50 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 52 ][ 1 ] - sd.cut_case_offsets[ 52 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 52 ][ 4 ] - sd.cut_case_offsets[ 52 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 56 ][ 1 ] - sd.cut_case_offsets[ 56 ][ 0 ] ) * 1
    );

    fc( s6(),
        ( sd.cut_case_offsets[ 0 ][ 1 ] - sd.cut_case_offsets[ 0 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 3 ][ 1 ] - sd.cut_case_offsets[ 3 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 6 ][ 1 ] - sd.cut_case_offsets[ 6 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 12 ][ 1 ] - sd.cut_case_offsets[ 12 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 2 ] - sd.cut_case_offsets[ 21 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 3 ] - sd.cut_case_offsets[ 21 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 5 ] - sd.cut_case_offsets[ 21 ][ 4 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 6 ] - sd.cut_case_offsets[ 21 ][ 5 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 7 ] - sd.cut_case_offsets[ 21 ][ 6 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 8 ] - sd.cut_case_offsets[ 21 ][ 7 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 9 ] - sd.cut_case_offsets[ 21 ][ 8 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 10 ] - sd.cut_case_offsets[ 21 ][ 9 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 12 ] - sd.cut_case_offsets[ 21 ][ 11 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 13 ] - sd.cut_case_offsets[ 21 ][ 12 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 15 ] - sd.cut_case_offsets[ 21 ][ 14 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 16 ] - sd.cut_case_offsets[ 21 ][ 15 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 17 ] - sd.cut_case_offsets[ 21 ][ 16 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 18 ] - sd.cut_case_offsets[ 21 ][ 17 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 19 ] - sd.cut_case_offsets[ 21 ][ 18 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 20 ] - sd.cut_case_offsets[ 21 ][ 19 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 21 ] - sd.cut_case_offsets[ 21 ][ 20 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 22 ] - sd.cut_case_offsets[ 21 ][ 21 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 23 ] - sd.cut_case_offsets[ 21 ][ 22 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 25 ] - sd.cut_case_offsets[ 21 ][ 24 ] ) * 1 +
        ( sd.cut_case_offsets[ 21 ][ 26 ] - sd.cut_case_offsets[ 21 ][ 25 ] ) * 1 +
        ( sd.cut_case_offsets[ 23 ][ 2 ] - sd.cut_case_offsets[ 23 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 23 ][ 3 ] - sd.cut_case_offsets[ 23 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 24 ][ 1 ] - sd.cut_case_offsets[ 24 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 27 ][ 2 ] - sd.cut_case_offsets[ 27 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 27 ][ 3 ] - sd.cut_case_offsets[ 27 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 29 ][ 2 ] - sd.cut_case_offsets[ 29 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 29 ][ 3 ] - sd.cut_case_offsets[ 29 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 33 ][ 1 ] - sd.cut_case_offsets[ 33 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 1 ] - sd.cut_case_offsets[ 42 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 2 ] - sd.cut_case_offsets[ 42 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 3 ] - sd.cut_case_offsets[ 42 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 4 ] - sd.cut_case_offsets[ 42 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 5 ] - sd.cut_case_offsets[ 42 ][ 4 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 7 ] - sd.cut_case_offsets[ 42 ][ 6 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 8 ] - sd.cut_case_offsets[ 42 ][ 7 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 11 ] - sd.cut_case_offsets[ 42 ][ 10 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 12 ] - sd.cut_case_offsets[ 42 ][ 11 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 14 ] - sd.cut_case_offsets[ 42 ][ 13 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 15 ] - sd.cut_case_offsets[ 42 ][ 14 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 16 ] - sd.cut_case_offsets[ 42 ][ 15 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 17 ] - sd.cut_case_offsets[ 42 ][ 16 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 18 ] - sd.cut_case_offsets[ 42 ][ 17 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 19 ] - sd.cut_case_offsets[ 42 ][ 18 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 21 ] - sd.cut_case_offsets[ 42 ][ 20 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 22 ] - sd.cut_case_offsets[ 42 ][ 21 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 24 ] - sd.cut_case_offsets[ 42 ][ 23 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 25 ] - sd.cut_case_offsets[ 42 ][ 24 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 26 ] - sd.cut_case_offsets[ 42 ][ 25 ] ) * 1 +
        ( sd.cut_case_offsets[ 42 ][ 27 ] - sd.cut_case_offsets[ 42 ][ 26 ] ) * 1 +
        ( sd.cut_case_offsets[ 43 ][ 2 ] - sd.cut_case_offsets[ 43 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 43 ][ 3 ] - sd.cut_case_offsets[ 43 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 45 ][ 2 ] - sd.cut_case_offsets[ 45 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 45 ][ 3 ] - sd.cut_case_offsets[ 45 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 46 ][ 1 ] - sd.cut_case_offsets[ 46 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 46 ][ 4 ] - sd.cut_case_offsets[ 46 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 48 ][ 1 ] - sd.cut_case_offsets[ 48 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 53 ][ 2 ] - sd.cut_case_offsets[ 53 ][ 1 ] ) * 1 +
        ( sd.cut_case_offsets[ 53 ][ 3 ] - sd.cut_case_offsets[ 53 ][ 2 ] ) * 1 +
        ( sd.cut_case_offsets[ 54 ][ 1 ] - sd.cut_case_offsets[ 54 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 54 ][ 4 ] - sd.cut_case_offsets[ 54 ][ 3 ] ) * 1 +
        ( sd.cut_case_offsets[ 58 ][ 1 ] - sd.cut_case_offsets[ 58 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 58 ][ 4 ] - sd.cut_case_offsets[ 58 ][ 3 ] ) * 1
    );
    ks->free_TF( score_best_sub_case );
    ks->free_TI( index_best_sub_case );
}


// =======================================================================================
ShapeType *s6() {
    static S6 res;
    return &res;
}

} // namespace sdot
