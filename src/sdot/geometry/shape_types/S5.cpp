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
    virtual void        cut_count  ( const std::function<void(const ShapeType *,BI)> &fc, const BI **offsets ) const override;
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

void S5::cut_count( const std::function<void(const ShapeType *,BI)> &fc, const BI **offsets ) const {
    fc( this,
        ( offsets[ 1 ][ 0 ] - offsets[ 0 ][ 0 ] ) * 1 +
        ( offsets[ 1 ][ 1 ] - offsets[ 0 ][ 1 ] ) * 0 +
        ( offsets[ 1 ][ 2 ] - offsets[ 0 ][ 2 ] ) * 0 +
        ( offsets[ 1 ][ 3 ] - offsets[ 0 ][ 3 ] ) * 0 +
        ( offsets[ 1 ][ 4 ] - offsets[ 0 ][ 4 ] ) * 0 +
        ( offsets[ 1 ][ 5 ] - offsets[ 0 ][ 5 ] ) * 0 +
        ( offsets[ 1 ][ 6 ] - offsets[ 0 ][ 6 ] ) * 0 +
        ( offsets[ 1 ][ 7 ] - offsets[ 0 ][ 7 ] ) * 0 +
        ( offsets[ 1 ][ 8 ] - offsets[ 0 ][ 8 ] ) * 0 +
        ( offsets[ 1 ][ 9 ] - offsets[ 0 ][ 9 ] ) * 0 +
        ( offsets[ 1 ][ 10 ] - offsets[ 0 ][ 10 ] ) * 0 +
        ( offsets[ 1 ][ 11 ] - offsets[ 0 ][ 11 ] ) * 0 +
        ( offsets[ 1 ][ 12 ] - offsets[ 0 ][ 12 ] ) * 0 +
        ( offsets[ 1 ][ 13 ] - offsets[ 0 ][ 13 ] ) * 0 +
        ( offsets[ 1 ][ 14 ] - offsets[ 0 ][ 14 ] ) * 0 +
        ( offsets[ 1 ][ 15 ] - offsets[ 0 ][ 15 ] ) * 0 +
        ( offsets[ 1 ][ 16 ] - offsets[ 0 ][ 16 ] ) * 0 +
        ( offsets[ 1 ][ 17 ] - offsets[ 0 ][ 17 ] ) * 0 +
        ( offsets[ 1 ][ 18 ] - offsets[ 0 ][ 18 ] ) * 0 +
        ( offsets[ 1 ][ 19 ] - offsets[ 0 ][ 19 ] ) * 0 +
        ( offsets[ 1 ][ 20 ] - offsets[ 0 ][ 20 ] ) * 0 +
        ( offsets[ 1 ][ 21 ] - offsets[ 0 ][ 21 ] ) * 0 +
        ( offsets[ 1 ][ 22 ] - offsets[ 0 ][ 22 ] ) * 0 +
        ( offsets[ 1 ][ 23 ] - offsets[ 0 ][ 23 ] ) * 0 +
        ( offsets[ 1 ][ 24 ] - offsets[ 0 ][ 24 ] ) * 0 +
        ( offsets[ 1 ][ 25 ] - offsets[ 0 ][ 25 ] ) * 0 +
        ( offsets[ 1 ][ 26 ] - offsets[ 0 ][ 26 ] ) * 0 +
        ( offsets[ 1 ][ 27 ] - offsets[ 0 ][ 27 ] ) * 0 +
        ( offsets[ 1 ][ 28 ] - offsets[ 0 ][ 28 ] ) * 0 +
        ( offsets[ 1 ][ 29 ] - offsets[ 0 ][ 29 ] ) * 0 +
        ( offsets[ 1 ][ 30 ] - offsets[ 0 ][ 30 ] ) * 0 +
        ( offsets[ 1 ][ 31 ] - offsets[ 0 ][ 31 ] ) * 0
    );
}


// =======================================================================================
ShapeType *s5() {
    static S5 res;
    return &res;
}

} // namespace sdot
