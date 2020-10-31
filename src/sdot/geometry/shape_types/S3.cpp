// generated file
#include "../ShapeData.h"
#include "../VtkOutput.h"
#include "S3.h"
#include "S4.h"

namespace sdot {

class S3 : public ShapeType {
public:
    virtual void        display_vtk( VtkOutput &vo, const double **tfs, const BI **tis, unsigned dim, BI nb_items, VtkOutput::Pt *offsets ) const override;
    virtual void        cut_rese   ( const std::function<void(const ShapeType *,BI)> &fc, const BI *cut_case_offsets ) const override;
    virtual unsigned    nb_nodes   () const override { return 3; }
    virtual unsigned    nb_faces   () const override { return 3; }
    virtual void        cut_ops    ( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const override;
    virtual std::string name       () const override { return "S3"; }
};

void S3::cut_ops( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const {
    ShapeData &nsd_S3 = new_shape_map.find( s3() )->second;
    ShapeData &nsd_S4 = new_shape_map.find( s4() )->second;

    ks->mk_items_n3_0_0_1_1_2_2_f3_0_1_2( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, { 0, 1, 2 }, 0, cut_ids, N<2>() );
    ks->mk_items_n4_0_1_1_1_2_2_2_0_f4_0_1_2_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2 }, { 0, 1, 2 }, 1, cut_ids, N<2>() );
    ks->mk_items_n4_1_2_2_2_0_0_0_1_f4_1_2_0_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2 }, { 0, 1, 2 }, 2, cut_ids, N<2>() );
    ks->mk_items_n3_1_2_2_2_2_0_f3_1_2_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, { 0, 1, 2 }, 3, cut_ids, N<2>() );
    ks->mk_items_n4_2_0_0_0_1_1_1_2_f4_2_0_1_c( nsd_S4, { 0, 1, 2, 3 }, { 0, 1, 2, 3 }, old_shape_data, { 0, 1, 2 }, { 0, 1, 2 }, 4, cut_ids, N<2>() );
    ks->mk_items_n3_0_1_1_1_1_2_f3_0_1_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, { 0, 1 }, 5, cut_ids, N<2>() );
    ks->mk_items_n3_2_0_0_0_0_1_f3_2_0_c( nsd_S3, { 0, 1, 2 }, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, { 0, 1, 2 }, 6, cut_ids, N<2>() );
}

void S3::display_vtk( VtkOutput &vo, const double **tfs, const BI **tis, unsigned /*dim*/, BI nb_items, VtkOutput::Pt *offsets ) const {
    using Pt = VtkOutput::Pt;
    if ( offsets ) {
        for( BI i = 0; i < nb_items; ++i ) {
            vo.add_triangle( {
                 Pt{ tfs[ 0 ][ i ], tfs[ 1 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
                 Pt{ tfs[ 2 ][ i ], tfs[ 3 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
                 Pt{ tfs[ 4 ][ i ], tfs[ 5 ][ i ], 0.0 } + offsets[ tis[ 0 ][ i ] ],
            } );
        }
    } else {
        for( BI i = 0; i < nb_items; ++i ) {
            vo.add_triangle( {
                 Pt{ tfs[ 0 ][ i ], tfs[ 1 ][ i ], 0.0 },
                 Pt{ tfs[ 2 ][ i ], tfs[ 3 ][ i ], 0.0 },
                 Pt{ tfs[ 4 ][ i ], tfs[ 5 ][ i ], 0.0 },
            } );
        }
    }
}

void S3::cut_rese( const std::function<void(const ShapeType *,BI)> &fc, const BI *cut_case_offsets ) const {
    fc( s3(),
        ( cut_case_offsets[ 1 ] - cut_case_offsets[ 0 ] ) * 1 +
        ( cut_case_offsets[ 4 ] - cut_case_offsets[ 3 ] ) * 1 +
        ( cut_case_offsets[ 6 ] - cut_case_offsets[ 5 ] ) * 1 +
        ( cut_case_offsets[ 7 ] - cut_case_offsets[ 6 ] ) * 1
    );
    fc( s4(),
        ( cut_case_offsets[ 2 ] - cut_case_offsets[ 1 ] ) * 1 +
        ( cut_case_offsets[ 3 ] - cut_case_offsets[ 2 ] ) * 1 +
        ( cut_case_offsets[ 5 ] - cut_case_offsets[ 4 ] ) * 1
    );
}



// =======================================================================================
ShapeType *s3() {
    static S3 res;
    return &res;
}

} // namespace sdot
