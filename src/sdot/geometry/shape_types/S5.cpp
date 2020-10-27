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
    virtual void        cut_count  ( const std::function<void(const ShapeType *,BI)> &fc, const BI *count_by_case ) const override;
    virtual unsigned    nb_nodes   () const override { return 5; }
    virtual unsigned    nb_faces   () const override { return 5; }
    virtual std::string name       () const override { return "S5"; }

    virtual void        cut_ops    ( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI /*dim*/ ) const override {
        ShapeData &nsd_S5 = new_shape_map.find( s5() )->second;

        ks->mk_items_0_0_1_1_2_2_3_3_4_4( nsd_S5, { 0, 1, 2, 3, 4 }, old_shape_data, { 0, 1, 2, 3, 4 }, 0, cut_ids, N<2>() );
    }
};

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

void S5::cut_count( const std::function<void(const ShapeType *,BI)> &fc, const BI *count_by_case ) const {
    fc( this,
        count_by_case[ 0 ] * 1 +
        count_by_case[ 1 ] * 0 +
        count_by_case[ 2 ] * 0 +
        count_by_case[ 3 ] * 0 +
        count_by_case[ 4 ] * 0 +
        count_by_case[ 5 ] * 0 +
        count_by_case[ 6 ] * 0 +
        count_by_case[ 7 ] * 0 +
        count_by_case[ 8 ] * 0 +
        count_by_case[ 9 ] * 0 +
        count_by_case[ 10 ] * 0 +
        count_by_case[ 11 ] * 0 +
        count_by_case[ 12 ] * 0 +
        count_by_case[ 13 ] * 0 +
        count_by_case[ 14 ] * 0 +
        count_by_case[ 15 ] * 0 +
        count_by_case[ 16 ] * 0 +
        count_by_case[ 17 ] * 0 +
        count_by_case[ 18 ] * 0 +
        count_by_case[ 19 ] * 0 +
        count_by_case[ 20 ] * 0 +
        count_by_case[ 21 ] * 0 +
        count_by_case[ 22 ] * 0 +
        count_by_case[ 23 ] * 0 +
        count_by_case[ 24 ] * 0 +
        count_by_case[ 25 ] * 0 +
        count_by_case[ 26 ] * 0 +
        count_by_case[ 27 ] * 0 +
        count_by_case[ 28 ] * 0 +
        count_by_case[ 29 ] * 0 +
        count_by_case[ 30 ] * 0 +
        count_by_case[ 31 ] * 0
    );
}


// =======================================================================================
ShapeType *s5() {
    static S5 res;
    return &res;
}

} // namespace sdot
