// generated file
#include "../ShapeData.h"
#include "../VtkOutput.h"
#include "S3.h"
#include "S4.h"
#include "S5.h"

namespace sdot {

class S3 : public ShapeType {
public:
    virtual void        display_vtk( VtkOutput &vo, const double **tfs, const BI **tis, unsigned dim, BI nb_items ) const override;
    virtual void        cut_count  ( const std::function<void(const ShapeType *,BI)> &fc, const BI *count_by_case ) const override;
    virtual unsigned    nb_nodes   () const override { return 3; }
    virtual unsigned    nb_faces   () const override { return 3; }
    virtual std::string name       () const override { return "S3"; }

    virtual void        cut_ops    ( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const override {
        ShapeData &nsd_S3 = new_shape_map.find( s3() )->second;

        ks->mk_items_0_0_1_1_2_2( nsd_S3, { 0, 1, 2 }, old_shape_data, { 0, 1, 2 }, 0, cut_ids, N<2>() );
    }
};

void S3::display_vtk( VtkOutput &vo, const double **tfs, const BI **/*tis*/, unsigned /*dim*/, BI nb_items ) const {
    using Pt = VtkOutput::Pt;
    for( BI i = 0; i < nb_items; ++i ) {
        vo.add_triangle( {
             Pt{ tfs[ 0 ][ i ], tfs[ 1 ][ i ], 0.0 },
             Pt{ tfs[ 2 ][ i ], tfs[ 3 ][ i ], 0.0 },
             Pt{ tfs[ 4 ][ i ], tfs[ 5 ][ i ], 0.0 },
        } );
    }
}

void S3::cut_count( const std::function<void(const ShapeType *,BI)> &fc, const BI *count_by_case ) const {
    fc( this,
        count_by_case[ 0 ] * 1 +
        count_by_case[ 1 ] * 0 +
        count_by_case[ 2 ] * 0 +
        count_by_case[ 3 ] * 0 +
        count_by_case[ 4 ] * 0 +
        count_by_case[ 5 ] * 0 +
        count_by_case[ 6 ] * 0 +
        count_by_case[ 7 ] * 0
    );
}


// =======================================================================================
ShapeType *s3() {
    static S3 res;
    return &res;
}

} // namespace sdot
