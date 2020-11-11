// generated file
#include "../ShapeData.h"
#include "../VtkOutput.h"
#include <iostream>
#include "S3.h"
#include "S4.h"

namespace sdot {

class S3 : public ShapeType {
public:
    virtual parex::Vec<TI> *cut_poss_count() const override;
    virtual CRN            *cut_rese_new  () const override;
    virtual void            display_vtk   ( const std::function<void( unsigned vtk_id, const parex::Vec<unsigned> &nodes )> &f ) const override;
    //    virtual void      cut_rese      ( const std::function<void(const ShapeType *,BI)> &fc, KernelSlot *ks, const ShapeData &sd ) const override;
    virtual unsigned        nb_nodes      () const override { return 3; }
    virtual unsigned        nb_faces      () const override { return 3; }
    //    virtual void      cut_ops       ( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const override;
    virtual std::string     name          () const override { return "S3"; }
};


parex::Vec<ShapeType::TI> *S3::cut_poss_count() const {
    static parex::Vec<TI> res{ 1, 1, 1, 1, 1, 1, 1, 0 };
    return &res;
}
//    fc( s3(),
//        ( sd.cut_case_offsets[ 0 ][ 1 ] - sd.cut_case_offsets[ 0 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 3 ][ 1 ] - sd.cut_case_offsets[ 3 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 5 ][ 1 ] - sd.cut_case_offsets[ 5 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 6 ][ 1 ] - sd.cut_case_offsets[ 6 ][ 0 ] ) * 1
//    );

//    fc( s4(),
//        ( sd.cut_case_offsets[ 1 ][ 1 ] - sd.cut_case_offsets[ 1 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 2 ][ 1 ] - sd.cut_case_offsets[ 2 ][ 0 ] ) * 1 +
//        ( sd.cut_case_offsets[ 4 ][ 1 ] - sd.cut_case_offsets[ 4 ][ 0 ] ) * 1
//    );

ShapeType::CRN *S3::cut_rese_new() const {
    static CRN res{
        { "S3", { 1, 0, 0, 1, 0, 1, 1, 0 } },
        { "S4", { 0, 1, 1, 0, 1, 0, 0, 0 } }
    };
    return &res;
}

//void S3::cut_ops( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const {
//    ShapeData &nsd_S3 = new_shape_map.find( s3() )->second;
//    ShapeData &nsd_S4 = new_shape_map.find( s4() )->second;

//    ks->mk_items_n3_0_0_1_1_2_2_f3_0_1_2( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data.cut_case_offsets[ 0 ][ 0 ], old_shape_data.cut_case_offsets[ 0 ][ 1 ], cut_ids, N<2>() );
//    ks->mk_items_n4_0_1_1_1_2_2_0_2_f4_0_1_2_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data.cut_case_offsets[ 1 ][ 0 ], old_shape_data.cut_case_offsets[ 1 ][ 1 ], cut_ids, N<2>() );
//    ks->mk_items_n4_0_0_0_1_1_2_2_2_f4_0_c_1_2( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data.cut_case_offsets[ 2 ][ 0 ], old_shape_data.cut_case_offsets[ 2 ][ 1 ], cut_ids, N<2>() );
//    ks->mk_items_n3_0_2_1_2_2_2_f3_c_1_2( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data.cut_case_offsets[ 3 ][ 0 ], old_shape_data.cut_case_offsets[ 3 ][ 1 ], cut_ids, N<2>() );
//    ks->mk_items_n4_0_0_1_1_1_2_0_2_f4_0_1_c_2( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data.cut_case_offsets[ 4 ][ 0 ], old_shape_data.cut_case_offsets[ 4 ][ 1 ], cut_ids, N<2>() );
//    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, { 0, 1 }, old_shape_data.cut_case_offsets[ 5 ][ 0 ], old_shape_data.cut_case_offsets[ 5 ][ 1 ], cut_ids, N<2>() );
//    ks->mk_items_n3_0_0_0_1_0_2_f3_0_c_2( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data.cut_case_offsets[ 6 ][ 0 ], old_shape_data.cut_case_offsets[ 6 ][ 1 ], cut_ids, N<2>() );
//}

void S3::display_vtk( const std::function<void( unsigned vtk_id, const parex::Vec<unsigned> &nodes )> &f ) const {
    f( 5, { 0, 1, 2 } );
}


// =======================================================================================
ShapeType *s3() {
    static S3 res;
    return &res;
}

} // namespace sdot
