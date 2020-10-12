#include "../VtkOutput.h"
#include "Triangle.h"

namespace sdot {

///
class Triangle : public ShapeType {
public:
    virtual void        display_vtk( VtkOutput &vo, const double **tfs, const BI **tis, unsigned dim, BI nb_items ) override;
    virtual void        cut_count  ( const std::function<void(std::string,BI)> &fc, const BI **offsets ) override;
    virtual unsigned    nb_nodes   () const override { return 3; }
    virtual unsigned    nb_faces   () const override { return 3; }
    virtual void        cut_ops    () const override {}
    virtual std::string name       () const override { return "triangle"; }
};

void Triangle::display_vtk( VtkOutput &vo, const double **tfs, const BI **/*tis*/, unsigned /*dim*/, BI nb_items ) {
    using Pt = VtkOutput::Pt;
    for( BI i = 0; i < nb_items; ++i ) {
        vo.add_triangle( {
             Pt{ tfs[ 0 ][ i ], tfs[ 1 ][ i ], 0.0 },
             Pt{ tfs[ 2 ][ i ], tfs[ 3 ][ i ], 0.0 },
             Pt{ tfs[ 4 ][ i ], tfs[ 5 ][ i ], 0.0 }
        } );
    }
}

void Triangle::cut_count( const std::function<void(std::string,BI)> &fc, const BI **offsets ) {
    fc( "triangle",
        ( offsets[ 1 ][ 0 ] - offsets[ 0 ][ 0 ] ) * 1 +
        ( offsets[ 1 ][ 1 ] - offsets[ 0 ][ 1 ] ) * 1 +
        ( offsets[ 1 ][ 2 ] - offsets[ 0 ][ 2 ] ) * 2 +
        ( offsets[ 1 ][ 3 ] - offsets[ 0 ][ 3 ] ) * 2 +
        ( offsets[ 1 ][ 4 ] - offsets[ 0 ][ 4 ] ) * 1 +
        ( offsets[ 1 ][ 5 ] - offsets[ 0 ][ 5 ] ) * 2 +
        ( offsets[ 1 ][ 6 ] - offsets[ 0 ][ 6 ] ) * 1 +
        ( offsets[ 1 ][ 7 ] - offsets[ 0 ][ 7 ] ) * 0
    );
}

// =======================================================================================
ShapeType *triangle() {
    static Triangle res;
    return &res;
}

}

