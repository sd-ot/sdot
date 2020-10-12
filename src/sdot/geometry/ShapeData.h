#pragma once

#include "../kernels/VecTF.h"
#include "ShapeType.h"
#include <vector>

namespace sdot {

/**
*/
struct ShapeData {
    using              BI           = VecTF::BI;

    /**/               ShapeData    ( KernelSlot *ks, ShapeType *shape_type, unsigned dim );

    void               resize       ( BI new_size );
    BI                 size         () const;

    std::vector<VecTF> coordinates; ///< all the x for node 0, all the y for node 0, ... all the x for node 1, ...
    ShapeType*         shape_type;  ///<
    std::vector<VecTF> face_ids;    ///< all the ids for node 0, all the ids for node 1, ...
    BI                 nb_items;    ///<
    VecTF              ids;         ///<
};

}
