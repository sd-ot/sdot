// generated file
#include <parex/support/TODO.h>
#include "../ShapeData.h"
#include "../VtkOutput.h"
#include <iostream>
#include "S3.h"
#include "S4.h"

namespace sdot {

class S4 : public ShapeType {
public:
    virtual parex::Vec<TI> *cut_poss_count() const override;
    virtual CRN            *cut_rese_new  () const override;
    virtual void            display_vtk   ( const std::function<void( TI vtk_id, const parex::Vec<TI> &nodes )> &f ) const override;
    virtual unsigned        nb_nodes      () const override { return 4; }
    virtual unsigned        nb_faces      () const override { return 4; }
    virtual VecCutOp       *cut_ops       () const override;
    virtual std::string     name          () const override { return "S4"; }
};

parex::Vec<ShapeType::TI> *S4::cut_poss_count() const {
    static parex::Vec<TI> res{ 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 0 };
    return &res;
}

ShapeType::CRN *S4::cut_rese_new() const {
    static CRN res{
        { s3(), { 1, 0, 0, 1, 0, 1, 1, 0 } },
        { s4(), { 0, 1, 1, 0, 1, 0, 0, 0 } }
    };
    TODO;
    return &res;
}

ShapeType::VecCutOp *S4::cut_ops() const {
    TODO;
    return nullptr;
    //    ShapeData &nsd_S3 = new_shape_map.find( s3() )->second;
    //    ShapeData &nsd_S4 = new_shape_map.find( s4() )->second;
    //    ShapeData &nsd_S5 = new_shape_map.find( s5() )->second;
    //    ShapeData &nsd_S6 = new_shape_map.find( s6() )->second;

    //    ks->mk_items_n4_0_0_1_1_2_2_3_3_f4_0_1_2_3( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 0 ][ 0 ], old_shape_data.cut_case_offsets[ 0 ][ 1 ], cut_ids, N<2>() );
    //    ks->mk_items_n5_0_1_1_1_2_2_3_3_0_3_f5_0_1_2_3_c( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 1 ][ 0 ], old_shape_data.cut_case_offsets[ 1 ][ 1 ], cut_ids, N<2>() );
    //    ks->mk_items_n5_0_0_0_1_1_2_2_2_3_3_f5_0_c_1_2_3( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 2 ][ 0 ], old_shape_data.cut_case_offsets[ 2 ][ 1 ], cut_ids, N<2>() );
    //    ks->mk_items_n4_0_3_1_2_2_2_3_3_f4_c_1_2_3( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 3 ][ 0 ], old_shape_data.cut_case_offsets[ 3 ][ 1 ], cut_ids, N<2>() );
    //    ks->mk_items_n5_0_0_1_1_1_2_2_3_3_3_f5_0_1_c_2_3( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 4 ][ 0 ], old_shape_data.cut_case_offsets[ 4 ][ 1 ], cut_ids, N<2>() );
    //    ks->mk_items_n3_0_3_2_3_3_3_f3_c_2_3_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 5 ][ 0 ], old_shape_data.cut_case_offsets[ 5 ][ 1 ], cut_ids, N<2>() );
    //    ks->mk_items_n6_0_1_1_1_1_2_2_3_3_3_0_3_f6_0_1_c_2_3_c( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 5 ][ 1 ], old_shape_data.cut_case_offsets[ 5 ][ 2 ], cut_ids, N<2>() );
    //    ks->mk_items_n4_0_0_0_1_2_3_3_3_f4_0_c_2_3( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 6 ][ 0 ], old_shape_data.cut_case_offsets[ 6 ][ 1 ], cut_ids, N<2>() );
    //    ks->mk_items_n3_0_3_2_3_3_3_f3_c_2_3( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 7 ][ 0 ], old_shape_data.cut_case_offsets[ 7 ][ 1 ], cut_ids, N<2>() );
    //    ks->mk_items_n5_0_0_1_1_2_2_2_3_0_3_f5_0_1_2_c_3( nsd_S5, { 0, 1, 2, 3, 4 }, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 8 ][ 0 ], old_shape_data.cut_case_offsets[ 8 ][ 1 ], cut_ids, N<2>() );
    //    ks->mk_items_n4_0_1_1_1_2_2_2_3_f4_0_1_2_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2 }, old_shape_data.cut_case_offsets[ 9 ][ 0 ], old_shape_data.cut_case_offsets[ 9 ][ 1 ], cut_ids, N<2>() );
    //    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n3_0_0_0_1_0_3_f3_0_c_3( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 10 ][ 0 ], old_shape_data.cut_case_offsets[ 10 ][ 1 ], cut_ids, N<2>() );
    //    ks->mk_items_n6_0_0_0_1_1_2_2_2_2_3_0_3_f6_0_c_1_2_c_3( nsd_S6, { 0, 1, 2, 3, 4, 5 }, { 0, 1, 2, 3, 4, 5 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 10 ][ 1 ], old_shape_data.cut_case_offsets[ 10 ][ 2 ], cut_ids, N<2>() );
    //    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2 }, old_shape_data.cut_case_offsets[ 11 ][ 0 ], old_shape_data.cut_case_offsets[ 11 ][ 1 ], cut_ids, N<2>() );
    //    ks->mk_items_n4_0_0_1_1_1_2_0_3_f4_0_1_c_3( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 12 ][ 0 ], old_shape_data.cut_case_offsets[ 12 ][ 1 ], cut_ids, N<2>() );
    //    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, { 0, 1 }, old_shape_data.cut_case_offsets[ 13 ][ 0 ], old_shape_data.cut_case_offsets[ 13 ][ 1 ], cut_ids, N<2>() );
    //    ks->mk_items_n3_0_0_0_1_0_3_f3_0_c_3( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 14 ][ 0 ], old_shape_data.cut_case_offsets[ 14 ][ 1 ], cut_ids, N<2>() );
}

void S4::display_vtk( const std::function<void( TI vtk_id, const parex::Vec<TI> &nodes )> &f ) const {
    f( 9, { 0, 1, 2, 3 } );
}


//void S4::cut_rese( const std::function<void(const ShapeType *,BI)> &fc, KernelSlot *ks, const ShapeData &sd ) const {
//    BI max_nb_item_with_sub_case = 0;
//    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] );
//    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] );

//    void *score_best_sub_case = ks->allocate_TF( max_nb_item_with_sub_case );
//    void *index_best_sub_case = ks->allocate_TI( max_nb_item_with_sub_case );

//    if ( sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] ) {
//        static std::vector<BI> nv{
//            0, 3, 2, 3,
//            1, 2, 0, 1,
//            1, 2, 2, 3,
//            0, 3, 0, 1,
//        };

//        VecTI nn{ ks, nv };
//        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] );

//        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 5 ][ 0 ], sd.cut_case_offsets[ 5 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
//        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 5 ][ 0 ], sd.cut_case_offsets[ 5 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );

//        ks->sort_TI_in_range( sd.cut_case_offsets[ 5 ].data(), index_best_sub_case, sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ], 2, sd.cut_indices, sd.cut_case_offsets[ 5 ][ 0 ] );
//    }

//    if ( sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] ) {
//        static std::vector<BI> nv{
//            2, 3, 1, 2,
//            0, 1, 0, 3,
//            0, 1, 1, 2,
//            2, 3, 0, 3,
//        };

//        VecTI nn{ ks, nv };
//        ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] );

//        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 10 ][ 0 ], sd.cut_case_offsets[ 10 ][ 1 ], 0, nn.data(), 0, 2, N<2>() );
//        ks->update_scores( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 10 ][ 0 ], sd.cut_case_offsets[ 10 ][ 1 ], 1, nn.data(), 8, 2, N<2>() );

//        ks->sort_TI_in_range( sd.cut_case_offsets[ 10 ].data(), index_best_sub_case, sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ], 2, sd.cut_indices, sd.cut_case_offsets[ 10 ][ 0 ] );
//    }

//    fc( s3(),
//        ( sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] ) * 2 +
//        ( sd.cut_case_offsets[ 7 ][ 1 ] - sd.cut_case_offsets[ 7 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] ) * 2 +
//        ( sd.cut_case_offsets[ 11 ][ 1 ] - sd.cut_case_offsets[ 11 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 13 ][ 1 ] - sd.cut_case_offsets[ 13 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 14 ][ 1 ] - sd.cut_case_offsets[ 14 ][ 0 ] ) * 1
//    );

//    fc( s4(),
//        ( sd.cut_case_offsets[ 0 ][ 1 ] - sd.cut_case_offsets[ 0 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 3 ][ 1 ] - sd.cut_case_offsets[ 3 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 6 ][ 1 ] - sd.cut_case_offsets[ 6 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 9 ][ 1 ] - sd.cut_case_offsets[ 9 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 12 ][ 1 ] - sd.cut_case_offsets[ 12 ][ 0 ] ) * 1
//    );

//    fc( s5(),
//        ( sd.cut_case_offsets[ 1 ][ 1 ] - sd.cut_case_offsets[ 1 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 2 ][ 1 ] - sd.cut_case_offsets[ 2 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 4 ][ 1 ] - sd.cut_case_offsets[ 4 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 8 ][ 1 ] - sd.cut_case_offsets[ 8 ][ 0 ] ) * 1
//    );

//    fc( s6(),
//        ( sd.cut_case_offsets[ 5 ][ 2 ] - sd.cut_case_offsets[ 5 ][ 1 ] ) * 1 +
//        ( sd.cut_case_offsets[ 10 ][ 2 ] - sd.cut_case_offsets[ 10 ][ 1 ] ) * 1
//    );

//    ks->free_TF( score_best_sub_case );
//    ks->free_TI( index_best_sub_case );
//}


// =======================================================================================
ShapeType *s4() {
    static S4 res;
    return &res;
}

} // namespace sdot
