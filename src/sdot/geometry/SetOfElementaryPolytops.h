#pragma once

#include "ElementaryPolytopTypes.h"

namespace sdot {

/**
*/
class SetOfElementaryPolytops {
public:
    using                  Value                  = parex::Value;
    using                  TI                     = std::size_t;

    /**/                   SetOfElementaryPolytops( const ElementaryPolytopTypes &ept, std::string scalar_type = "FP64", std::string index_type = "PI64" );

    void                   write_to_stream        ( std::ostream &os ) const;
    void                   display_vtk            ( const std::string &filename ) const;

    void                   add_repeated           ( const Value &shape_name, const Value &count, const Value &coordinates, const Value &face_ids = 0, const Value &beg_ids = 0 );
    void                   plane_cut              ( const Value &normals, const Value &scalar_products, const Value &cut_ids );

private:
    std::string            scalar_type;           ///<
    std::string            index_type;            ///<
    Value                  shape_map;             ///<
    ElementaryPolytopTypes ept;                   ///<
};

}
