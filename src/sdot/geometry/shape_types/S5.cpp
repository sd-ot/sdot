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
    virtual void        cut_count  ( const std::function<void(const ShapeType *,BI)> &fc, const BI *case_offsets ) const override;
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

void S5::cut_count( const std::function<void(const ShapeType *,BI)> &fc, const BI *case_offsets ) const {
    fc( this,
        ( case_offsets[ 1 ] - case_offsets[ 0 ] ) * 1 +
        ( case_offsets[ 2 ] - case_offsets[ 1 ] ) * 0 +
        ( case_offsets[ 3 ] - case_offsets[ 2 ] ) * 0 +
        ( case_offsets[ 4 ] - case_offsets[ 3 ] ) * 0 +
        ( case_offsets[ 5 ] - case_offsets[ 4 ] ) * 0 +
        ( case_offsets[ 6 ] - case_offsets[ 5 ] ) * 0 +
        ( case_offsets[ 7 ] - case_offsets[ 6 ] ) * 0 +
        ( case_offsets[ 8 ] - case_offsets[ 7 ] ) * 0 +
        ( case_offsets[ 9 ] - case_offsets[ 8 ] ) * 0 +
        ( case_offsets[ 10 ] - case_offsets[ 9 ] ) * 0 +
        ( case_offsets[ 11 ] - case_offsets[ 10 ] ) * 0 +
        ( case_offsets[ 12 ] - case_offsets[ 11 ] ) * 0 +
        ( case_offsets[ 13 ] - case_offsets[ 12 ] ) * 0 +
        ( case_offsets[ 14 ] - case_offsets[ 13 ] ) * 0 +
        ( case_offsets[ 15 ] - case_offsets[ 14 ] ) * 0 +
        ( case_offsets[ 16 ] - case_offsets[ 15 ] ) * 0 +
        ( case_offsets[ 17 ] - case_offsets[ 16 ] ) * 0 +
        ( case_offsets[ 18 ] - case_offsets[ 17 ] ) * 0 +
        ( case_offsets[ 19 ] - case_offsets[ 18 ] ) * 0 +
        ( case_offsets[ 20 ] - case_offsets[ 19 ] ) * 0 +
        ( case_offsets[ 21 ] - case_offsets[ 20 ] ) * 0 +
        ( case_offsets[ 22 ] - case_offsets[ 21 ] ) * 0 +
        ( case_offsets[ 23 ] - case_offsets[ 22 ] ) * 0 +
        ( case_offsets[ 24 ] - case_offsets[ 23 ] ) * 0 +
        ( case_offsets[ 25 ] - case_offsets[ 24 ] ) * 0 +
        ( case_offsets[ 26 ] - case_offsets[ 25 ] ) * 0 +
        ( case_offsets[ 27 ] - case_offsets[ 26 ] ) * 0 +
        ( case_offsets[ 28 ] - case_offsets[ 27 ] ) * 0 +
        ( case_offsets[ 29 ] - case_offsets[ 28 ] ) * 0 +
        ( case_offsets[ 30 ] - case_offsets[ 29 ] ) * 0 +
        ( case_offsets[ 31 ] - case_offsets[ 30 ] ) * 0 +
        ( case_offsets[ 32 ] - case_offsets[ 31 ] ) * 0
    );
}


// =======================================================================================
ShapeType *s5() {
    static S5 res;
    return &res;
}

} // namespace sdot
