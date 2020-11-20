#pragma once

#include <parex/Value.h>

namespace sdot {

/**
*/
class SetOfElementaryPolytops {
public:
    using                  TaskRef                = parex::TaskRef;
    using                  Value                  = parex::Value;
    using                  TI                     = std::size_t;

    /**/                   SetOfElementaryPolytops( const TaskRef &n_dim, const TaskRef &s_scalar_type, const TaskRef &s_index_type, std::string shape_list );
    /**/                   SetOfElementaryPolytops( int dim, std::string scalar_type = "FP64", std::string index_type = "PI64" );

    void                   write_to_stream        ( std::ostream &os ) const;
    void                   display_vtk            ( const Value &filename ) const;

    void                   add_repeated           ( const Value &shape_name, const Value &count, const Value &coordinates, const Value &face_ids = 0, const Value &beg_ids = 0 );
    void                   plane_cut              ( const Value &normals, const Value &scalar_products, const Value &cut_ids );

private:
    static std::string     default_shape_list     ( int dim );

    std::string            shape_list;
    Value                  shape_map;             ///<
};

}
