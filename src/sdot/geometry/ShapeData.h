#pragma once

#include "ShapeType.h"

namespace sdot {

/**
*/
class ShapeData {
public:
    using              BI           = ShapeType::BI;

    /**/               ShapeData    ( KernelSlot *ks, const ShapeType *shape_type, unsigned dim );
    /**/              ~ShapeData    ();

    void               reserve      ( BI new_size );
    void               resize       ( BI new_size );

    const ShapeType*   shape_type;  ///<
    BI                 log2_rese;   ///<
    BI                 rese;        ///<
    BI                 size;        ///<
    unsigned           dim;         ///<
    KernelSlot*        ks;          ///<

    void*              coordinates; ///< all the x for node 0, all the y for node 0, ... all the x for node 1, ...
    void*              face_ids;    ///< all the ids for node 0, all the ids for node 1, ...
    void*              ids;         ///<

    enum {             out_scps, cut_case, offsets, indices };
    mutable void*      tmp[ 4 ];    ///<
};

}
