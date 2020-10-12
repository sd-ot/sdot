#pragma once

#include "../kernels/KernelSlot.h"
#include "../kernels/VecTF.h"

#include "shape_types/Triangle.h"
#include "ShapeData.h"

#include <map>

namespace sdot {

/**
*/
class SetOfElementaryPolytops {
public:
    using       BI                     = std::uint64_t;

    /**/        SetOfElementaryPolytops( KernelSlot *ks, unsigned dim );

    void        write_to_stream        ( std::ostream &os ) const;
    void        add_repeated           ( ShapeType *shape_type, BI count, const VecTF &coordinates );

private:
    using       ShapeMap               = std::map<ShapeType *,ShapeData>;

    ShapeData*  shape_data             ( ShapeType *shape_type );

    ShapeMap    shape_map;             ///<
    unsigned    dim;                   ///<
    KernelSlot *ks;                    ///< where to execute the code
};

}
