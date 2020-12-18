#pragma once

#include "ElementaryPolytopTypeSet.h"

namespace sdot {

/**
*/
class SetOfElementaryPolytops {
public:
    struct                     CtorParameters         { const parex::String &scalar_type = "FP64"; const parex::String &index_type = "PI64"; parex::Memory *dst = nullptr; parex::Scalar dim = 0; };

    /**/                       SetOfElementaryPolytops( const ElementaryPolytopTypeSet &elementary_polytop_type_set, const CtorParameters &parameters );

    void                       write_to_stream        ( std::ostream &os ) const;
    void                       display_vtk            ( const parex::String &filename ) const;

    //void                     add_repeated           ( const parex::String &shape_name, const Value &count, const Value &coordinates, const Value &face_ids = 0, const Value &beg_ids = 0 );
    //void                     plane_cut              ( const parex::String &normals, const Value &scalar_products, const Value &cut_ids );

private:
    parex::Rc<parex::Variable> elem_info;             ///<
    parex::Rc<parex::Variable> shape_map;             ///<
};

}
