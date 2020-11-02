// generated file
#include "../ShapeData.h"
#include "../VtkOutput.h"
#include "S3.h"
#include "S4.h"

namespace sdot {

class S4 : public ShapeType {
public:
    virtual std::vector<BI> cut_poss_count() const override;
    virtual void            display_vtk   ( VtkOutput &vo, const double **tfs, const BI **tis, unsigned dim, BI nb_items, VtkOutput::Pt *offsets ) const override;
    virtual void            cut_rese      ( const std::function<void(const ShapeType *,BI)> &fc, KernelSlot *ks, const ShapeData &sd ) const override;
    virtual unsigned        nb_nodes      () const override { return 4; }
    virtual unsigned        nb_faces      () const override { return 4; }
    virtual void            cut_ops       ( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const override;
    virtual std::string     name          () const override { return "S4"; }
};


std::vector<ShapeType::BI> S4::cut_poss_count() const {
    return { 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 4, 1, 1, 1, 1, 0 };
}

void S4::cut_ops( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const {
    ShapeData &nsd_S3 = new_shape_map.find( s3() )->second;
    ShapeData &nsd_S4 = new_shape_map.find( s4() )->second;

    ks->mk_items_n4_0_0_1_1_2_2_3_3_f4_0_1_2_3( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 0 ][ 0 ], old_shape_data.cut_case_offsets[ 0 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_2_2_f3_0_1_i_n4_2_2_3_3_3_0_0_1_f4_2_3_c_i( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 1 ][ 0 ], old_shape_data.cut_case_offsets[ 1 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_3_3_f3_1_2_i_n4_3_3_0_0_0_1_1_2_f4_3_0_c_i( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 2 ][ 0 ], old_shape_data.cut_case_offsets[ 2 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_3_3_3_0_f4_1_2_3_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 3 ][ 0 ], old_shape_data.cut_case_offsets[ 3 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_0_0_f3_2_3_i_n4_0_0_1_1_1_2_2_3_f4_0_1_c_i( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 4 ][ 0 ], old_shape_data.cut_case_offsets[ 4 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_0_f3_2_3_c_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 5 ][ 0 ], old_shape_data.cut_case_offsets[ 5 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_1_2_2_3_f4_0_1_c_i_n4_2_3_3_3_3_0_0_1_f4_2_3_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 5 ][ 1 ], old_shape_data.cut_case_offsets[ 5 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n4_2_3_3_3_3_0_0_1_f4_2_3_c_i_n4_0_1_1_1_1_2_2_3_f4_0_1_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 5 ][ 2 ], old_shape_data.cut_case_offsets[ 5 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c_n3_2_3_3_3_3_0_f3_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 5 ][ 3 ], old_shape_data.cut_case_offsets[ 5 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n4_2_3_3_3_0_0_0_1_f4_2_3_0_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 6 ][ 0 ], old_shape_data.cut_case_offsets[ 6 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_2_3_3_3_3_0_f3_2_3_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 7 ][ 0 ], old_shape_data.cut_case_offsets[ 7 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_3_0_0_0_1_1_f3_3_0_i_n4_1_1_2_2_2_3_3_0_f4_1_2_c_i( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 8 ][ 0 ], old_shape_data.cut_case_offsets[ 8 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_2_2_2_3_f4_0_1_2_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2 }, old_shape_data.cut_case_offsets[ 9 ][ 0 ], old_shape_data.cut_case_offsets[ 9 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_3_0_0_0_0_1_1_2_f4_3_0_c_i_n4_1_2_2_2_2_3_3_0_f4_1_2_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 10 ][ 0 ], old_shape_data.cut_case_offsets[ 10 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n3_3_0_0_0_0_1_f3_3_0_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 10 ][ 1 ], old_shape_data.cut_case_offsets[ 10 ][ 2 ], cut_ids, N<2>() );
    ks->mk_items_n3_3_0_0_0_0_1_f3_3_0_c_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 10 ][ 2 ], old_shape_data.cut_case_offsets[ 10 ][ 3 ], cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_2_3_3_0_f4_1_2_c_i_n4_3_0_0_0_0_1_1_2_f4_3_0_c_i( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 10 ][ 3 ], old_shape_data.cut_case_offsets[ 10 ][ 4 ], cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_3_f3_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2 }, old_shape_data.cut_case_offsets[ 11 ][ 0 ], old_shape_data.cut_case_offsets[ 11 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n4_3_0_0_0_1_1_1_2_f4_3_0_1_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 12 ][ 0 ], old_shape_data.cut_case_offsets[ 12 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, { 0, 1 }, old_shape_data.cut_case_offsets[ 13 ][ 0 ], old_shape_data.cut_case_offsets[ 13 ][ 1 ], cut_ids, N<2>() );
    ks->mk_items_n3_3_0_0_0_0_1_f3_3_0_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data.cut_case_offsets[ 14 ][ 0 ], old_shape_data.cut_case_offsets[ 14 ][ 1 ], cut_ids, N<2>() );
}

void S4::display_vtk( VtkOutput &vo, const double **tfs, const BI **tis, unsigned /*dim*/, BI nb_items, VtkOutput::Pt *offsets ) const {
    using Pt = VtkOutput::Pt;
    if ( offsets ) {
        for( BI i = 0; i < nb_items; ++i ) {
            vo.add_quad( {
                 Pt{ tfs[ 0 ][ i ], tfs[ 1 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
                 Pt{ tfs[ 2 ][ i ], tfs[ 3 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
                 Pt{ tfs[ 4 ][ i ], tfs[ 5 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
                 Pt{ tfs[ 6 ][ i ], tfs[ 7 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
            } );
        }
    } else {
        for( BI i = 0; i < nb_items; ++i ) {
            vo.add_quad( {
                 Pt{ tfs[ 0 ][ i ], tfs[ 1 ][ i ], 0.0 },
                 Pt{ tfs[ 2 ][ i ], tfs[ 3 ][ i ], 0.0 },
                 Pt{ tfs[ 4 ][ i ], tfs[ 5 ][ i ], 0.0 },
                 Pt{ tfs[ 6 ][ i ], tfs[ 7 ][ i ], 0.0 },
            } );
        }
    }
}

void S4::cut_rese( const std::function<void(const ShapeType *,BI)> &fc, KernelSlot *ks, const ShapeData &sd ) const {
    BI max_nb_item_with_sub_case = 0;
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] );
    max_nb_item_with_sub_case = std::max( max_nb_item_with_sub_case, sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] );

    void *score_best_sub_case = ks->allocate_TF( max_nb_item_with_sub_case );
    void *index_best_sub_case = ks->allocate_TI( max_nb_item_with_sub_case );

    ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] );
    ks->update_score_3_0_2_2_1_2_0_0( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 5 ][ 0 ], sd.cut_case_offsets[ 5 ][ 1 ], 0 );
    ks->update_score_1_2_2_2_3_0_0_0( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 5 ][ 0 ], sd.cut_case_offsets[ 5 ][ 1 ], 1 );
    ks->update_score_3_0_0_0_1_2_2_2( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 5 ][ 0 ], sd.cut_case_offsets[ 5 ][ 1 ], 2 );
    ks->update_score_1_2_0_0_3_0_2_2( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 5 ][ 0 ], sd.cut_case_offsets[ 5 ][ 1 ], 3 );

    ks->assign_TF( score_best_sub_case, 0, 0.0, sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] );
    ks->update_score_0_1_1_1_2_3_3_3( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 10 ][ 0 ], sd.cut_case_offsets[ 10 ][ 1 ], 0 );
    ks->update_score_2_3_1_1_0_1_3_3( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 10 ][ 0 ], sd.cut_case_offsets[ 10 ][ 1 ], 1 );
    ks->update_score_0_1_3_3_2_3_1_1( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 10 ][ 0 ], sd.cut_case_offsets[ 10 ][ 1 ], 2 );
    ks->update_score_2_3_3_3_0_1_1_1( score_best_sub_case, index_best_sub_case, sd, sd.cut_case_offsets[ 10 ][ 0 ], sd.cut_case_offsets[ 10 ][ 1 ], 3 );

    fc( s3(),
        ( sd.cut_case_offsets[ 1 ][ 1 ] - sd.cut_case_offsets[ 1 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 2 ][ 1 ] - sd.cut_case_offsets[ 2 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 4 ][ 1 ] - sd.cut_case_offsets[ 4 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 5 ][ 4 ] - sd.cut_case_offsets[ 5 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 7 ][ 1 ] - sd.cut_case_offsets[ 7 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 8 ][ 1 ] - sd.cut_case_offsets[ 8 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 10 ][ 2 ] - sd.cut_case_offsets[ 10 ][ 1 ] ) * 2 +
        ( sd.cut_case_offsets[ 10 ][ 3 ] - sd.cut_case_offsets[ 10 ][ 2 ] ) * 2 +
        ( sd.cut_case_offsets[ 11 ][ 1 ] - sd.cut_case_offsets[ 11 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 13 ][ 1 ] - sd.cut_case_offsets[ 13 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 14 ][ 1 ] - sd.cut_case_offsets[ 14 ][ 0 ] ) * 1
    );

    fc( s4(),
        ( sd.cut_case_offsets[ 0 ][ 1 ] - sd.cut_case_offsets[ 0 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 1 ][ 1 ] - sd.cut_case_offsets[ 1 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 2 ][ 1 ] - sd.cut_case_offsets[ 2 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 3 ][ 1 ] - sd.cut_case_offsets[ 3 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 4 ][ 1 ] - sd.cut_case_offsets[ 4 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 5 ][ 2 ] - sd.cut_case_offsets[ 5 ][ 1 ] ) * 2 +
        ( sd.cut_case_offsets[ 5 ][ 3 ] - sd.cut_case_offsets[ 5 ][ 2 ] ) * 2 +
        ( sd.cut_case_offsets[ 6 ][ 1 ] - sd.cut_case_offsets[ 6 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 8 ][ 1 ] - sd.cut_case_offsets[ 8 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 9 ][ 1 ] - sd.cut_case_offsets[ 9 ][ 0 ] ) * 1 +
        ( sd.cut_case_offsets[ 10 ][ 1 ] - sd.cut_case_offsets[ 10 ][ 0 ] ) * 2 +
        ( sd.cut_case_offsets[ 10 ][ 4 ] - sd.cut_case_offsets[ 10 ][ 3 ] ) * 2 +
        ( sd.cut_case_offsets[ 12 ][ 1 ] - sd.cut_case_offsets[ 12 ][ 0 ] ) * 1
    );
    ks->free_TF( score_best_sub_case );
    ks->free_TI( index_best_sub_case );
}


// =======================================================================================
ShapeType *s4() {
    static S4 res;
    return &res;
}

} // namespace sdot
