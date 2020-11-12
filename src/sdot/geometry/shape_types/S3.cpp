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
    virtual void            display_vtk   ( const std::function<void( TI vtk_id, const parex::Vec<TI> &nodes )> &f ) const override;
    virtual unsigned        nb_nodes      () const override { return 3; }
    virtual unsigned        nb_faces      () const override { return 3; }
    virtual VecCutOp       *cut_ops       () const override;
    virtual std::string     name          () const override { return "S3"; }
};


parex::Vec<ShapeType::TI> *S3::cut_poss_count() const {
    static parex::Vec<TI> res{ 1, 1, 1, 1, 1, 1, 1, 0 };
    return &res;
}

ShapeType::CRN *S3::cut_rese_new() const {
    static CRN res{
        { s3(), { 1, 0, 0, 1, 0, 1, 1, 0 } },
        { s4(), { 0, 1, 1, 0, 1, 0, 0, 0 } }
    };
    return &res;
}

ShapeType::VecCutOp *S3::cut_ops() const {
    static VecCutOp res{
        CutOp{ "3 0 0 1 1 2 2 3 0 1 2"       , { OutCutOp{ s3(), { 0, 1, 2    }, { 0, 1, 2    } } }, { 0, 1, 2 }, { 0, 1, 2 }, 0, 0 },
        CutOp{ "4 0 1 1 1 2 2 0 2 4 0 1 2 -1", { OutCutOp{ s4(), { 0, 1, 2, 3 }, { 0, 1, 2, 3 } } }, { 0, 1, 2 }, { 0, 1, 2 }, 1, 0 },
        CutOp{ "4 0 0 0 1 1 2 2 2 4 0 -1 1 2", { OutCutOp{ s4(), { 0, 1, 2, 3 }, { 0, 1, 2, 3 } } }, { 0, 1, 2 }, { 0, 1, 2 }, 2, 0 },
        CutOp{ "3 0 2 1 2 2 2 3 -1 1 2"      , { OutCutOp{ s3(), { 0, 1, 2    }, { 0, 1, 2    } } }, { 0, 1, 2 }, { 0, 1, 2 }, 3, 0 },
        CutOp{ "4 0 0 1 1 1 2 0 2 4 0 1 -1 2", { OutCutOp{ s4(), { 0, 1, 2, 3 }, { 0, 1, 2, 3 } } }, { 0, 1, 2 }, { 0, 1, 2 }, 4, 0 },
        CutOp{ "3 0 1 1 1 1 2 3 0 1 -1"      , { OutCutOp{ s3(), { 0, 1, 2    }, { 0, 1, 2    } } }, { 0, 1, 2 }, { 0, 1, 2 }, 5, 0 },
        CutOp{ "3 0 0 0 1 0 2 3 0 -1 2"      , { OutCutOp{ s3(), { 0, 1, 2    }, { 0, 1, 2    } } }, { 0, 1, 2 }, { 0, 1, 2 }, 6, 0 },
    };
    return &res;
}

void S3::display_vtk( const std::function<void( TI vtk_id, const parex::Vec<TI> &nodes )> &f ) const {
    f( 5, { 0, 1, 2 } );
}


// =======================================================================================
ShapeType *s3() {
    static S3 res;
    return &res;
}

} // namespace sdot
