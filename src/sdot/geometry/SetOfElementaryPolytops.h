#pragma once

#include "../kernels/KernelSlot.h"
#include "../kernels/VecTF.h"
#include "../kernels/VecTI.h"

#include "shape_types/Triangle.h"
#include "ShapeData.h"
#include "VtkOutput.h"

#include <map>

namespace sdot {

/**
*/
class SetOfElementaryPolytops {
public:
    using       BI                     = std::uint64_t;

    /**/        SetOfElementaryPolytops( KernelSlot *ks, unsigned dim );

    void        add_repeated           ( ShapeType *shape_type, BI count, const VecTF &coordinates, const VecTI &face_ids, BI beg_ids = 0 );

    void        plane_cut              ( const std::vector<VecTF> &normals, const VecTF &scalar_products, const VecTI &cut_ids );

    void        write_to_stream        ( std::ostream &os, const std::string &sp = {} ) const;
    void        display_vtk            ( VtkOutput &vo, VtkOutput::Pt *offsets = nullptr ) const;

private:
    using       ShapeMap               = std::map<const ShapeType *,ShapeData>;

    ShapeData*  shape_data             ( const ShapeType *shape_type );

    ShapeMap    shape_map;             ///<
    unsigned    dim;                   ///<
    KernelSlot *ks;                    ///< where to execute the code
};

}
