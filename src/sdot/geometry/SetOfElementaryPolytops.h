#pragma once

#include "ElementaryPolytopInfoList.h"
#include <parex/containers/xtensor.h>
#include <parex/Value.h>

namespace sdot {

/**
*/
class SetOfElementaryPolytops {
public:
    using                  TI                     = std::size_t;

    /**/                   SetOfElementaryPolytops( const ElementaryPolytopInfoList &elementary_polytop_info, const Value &scalar_type = "FP64", const Value &index_type = "PI64", const Value &dim = 0 );

    void                   write_to_stream        ( std::ostream &os ) const;
    void                   display_vtk            ( const Value &filename ) const;

    void                   add_repeated           ( const Value &shape_name, const Value &count, const Value &coordinates, const Value &face_ids = 0, const Value &beg_ids = 0 );
    void                   plane_cut              ( const Value &normals, const Value &scalar_products, const Value &cut_ids );

private:
    Rc<Task>               shape_map;             ///<
};

}
