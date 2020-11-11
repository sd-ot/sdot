#pragma once

#include <parex/Value.h>
#include "VtkOutput.h"

namespace sdot {

/**
*/
class SetOfElementaryPolytops {
public:
    using       Value                  = parex::Value;
    using       TI                     = std::size_t;

    /**/        SetOfElementaryPolytops( unsigned dim, std::string scalar_type = "FP64", std::string index_type = "PI64" );

    void        add_repeated           ( const std::string &shape_name, const Value &count, const Value &coordinates, const Value &face_ids = 0, const Value &beg_ids = 0 );
    void        plane_cut              ( const Value &normals, const Value &scalar_products, const Value &cut_ids );

    void        write_to_stream        ( std::ostream &os, const std::string &sp = {} ) const;
    void        display_vtk            ( VtkOutput &vo, VtkOutput::Pt *offsets = nullptr ) const;

private:
    std::string scalar_type;           ///<
    std::string index_type;            ///<
    Value       shape_map;             ///<
    unsigned    dim;                   ///<
};

}
