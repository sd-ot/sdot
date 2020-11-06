#pragma once

#include "shape_types/Triangle.h"
#include "shape_types/Quad.h"
#include "ShapeData.h"
#include "VtkOutput.h"

#include <parex/Value.h>
#include <map>

namespace sdot {

/**
*/
class SetOfElementaryPolytops {
public:
    using       Value                  = parex::Value;
    using       TI                     = std::size_t;

    /**/        SetOfElementaryPolytops( unsigned dim );

    void        add_repeated           ( ShapeType *shape_type, const Value &count, const Value &coordinates, const Value &face_ids = 0, const Value &beg_ids = 0 );

    //    void        plane_cut              ( const Value &normals, const Value &scalar_products, const Value &cut_ids );

    void        write_to_stream        ( std::ostream &os, const std::string &sp = {} ) const;
    //    void        display_vtk            ( VtkOutput &vo, VtkOutput::Pt *offsets = nullptr ) const;

    //private:
    //    using       ShapeMap               = std::map<const ShapeType *,ShapeData>;

    //    ShapeData*  shape_data             ( const ShapeType *shape_type );

    //    ShapeMap    shape_map;             ///<
    unsigned    dim;                   ///<
};

}
