#pragma once

#include "ElementaryPolytopInfoList.h"
#include <parex/containers/xtensor.h>
#include <parex/MemoryCpu.h>
#include <parex/Value.h>

class ElementaryPolytopInfoListContent;

namespace sdot {

/**
*/
class SetOfElementaryPolytops {
public:
    struct                 Parm                   { const Value &scalar_type = "FP64"; const Value &index_type = "PI64"; Memory *dst = nullptr; const Value &dim = 0; };

    /**/                   SetOfElementaryPolytops( const ElementaryPolytopInfoList &elementary_polytop_info, const Parm &types );

    void                   write_to_stream        ( std::ostream &os ) const;
    void                   display_vtk            ( const Value &filename ) const;

    void                   add_repeated           ( const Value &shape_name, const Value &count, const Value &coordinates, const Value &face_ids = 0, const Value &beg_ids = 0 );
    void                   plane_cut              ( const Value &normals, const Value &scalar_products, const Value &cut_ids );

private:
    static Type*           shape_map_type         ( const std::string &type_name, const ElementaryPolytopInfoListContent *epil, Type *scalar_type, Type *index_type, Memory *dst, int dim );

    Rc<Task>               elem_info;             ///<
    Rc<Task>               shape_map;             ///<
};

}
