#pragma once

#include <parex/Value.h>

namespace sdot {

/**
*/
class SetOfElementaryPolytops {
public:
    using                  TI                     = std::size_t;

    /**/                   SetOfElementaryPolytops( Value dim, Value scalar_type = "FP64", Value index_type = "PI64", Value elem_shapes = "" );

    void                   write_to_stream        ( std::ostream &os ) const;
    void                   display_vtk            ( const Value &filename ) const;

    void                   add_repeated           ( const Value &shape_name, const Value &count, const Value &coordinates, const Value &face_ids = 0, const Value &beg_ids = 0 );
    void                   plane_cut              ( const Value &normals, const Value &scalar_products, const Value &cut_ids );

private:
    static std::string     default_shape_list     ( int dim );

    Value                  shape_map;             ///<
};

}
