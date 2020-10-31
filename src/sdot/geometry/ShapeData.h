#pragma once

#include "ShapeType.h"
#include <vector>

namespace sdot {

/**
*/
class ShapeData {
public:
    using                   BI                = ShapeType::BI;

    /**/                    ShapeData         ( KernelSlot *ks, const ShapeType *shape_type, unsigned dim );
    /**/                   ~ShapeData         ();

    void                    reserve           ( BI new_size );
    void                    resize            ( BI new_size );

    const ShapeType*        shape_type;       ///<
    BI                      log2_rese;        ///<
    BI                      rese;             ///<
    BI                      size;             ///<
    unsigned                dim;              ///<
    KernelSlot*             ks;               ///<

    void*                   coordinates;      ///< all the x for node 0, all the y for node 0, ... all the x for node 1, ...
    void*                   face_ids;         ///< all the ids for node 0, all the ids for node 1, ...
    void*                   ids;              ///<

    mutable std::vector<BI> cut_case_offsets; ///<
    mutable void*           cut_out_scps;     ///< scalar product, for each item and for each node
    mutable void*           cut_indices;      ///< list of item numbers (item index), sorted by cut case
};

}
