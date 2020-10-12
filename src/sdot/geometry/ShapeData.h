#pragma once

#include "ShapeType.h"

namespace sdot {

/**
*/
struct ShapeData {
    using              BI           = ShapeType::BI;

    /**/               ShapeData    ( KernelSlot *ks, ShapeType *shape_type, unsigned dim );
    /**/              ~ShapeData    ();

    void               reserve      ( BI new_size );
    void               resize       ( BI new_size );

    ShapeType*         shape_type;  ///<
    BI                 log2_rese;   ///<
    BI                 rese;        ///<
    BI                 size;        ///<
    unsigned           dim;         ///<
    KernelSlot*        ks;          ///<

    void*              coordinates; ///< all the x for node 0, all the y for node 0, ... all the x for node 1, ...
    void*              face_ids;    ///< all the ids for node 0, all the ids for node 1, ...
    void*              ids;         ///<

    void*              tmp_off_0;   ///<
    void*              tmp_off_1;   ///<
    void*              tmp_scps;    ///< scalar products
};

}
