#pragma once

#include "ElementaryPolytopTypeSet.h"
#include <parex/Vector.h>
#include <parex/Tensor.h>

namespace sdot {

/**
*/
class SetOfElementaryPolytops {
public:
    struct                     CtorParameters         { const parex::String &scalar_type = parex::TypeInfo<double>::name(); const parex::String &index_type = parex::TypeInfo<std::uint64_t>::name(); parex::Memory *dst = nullptr; parex::Scalar dim = 0; };

    /**/                       SetOfElementaryPolytops( const ElementaryPolytopTypeSet &elementary_polytop_type_set, const CtorParameters &parameters );

    void                       write_to_stream        ( std::ostream &os ) const;
    void                       display_vtk            ( const parex::String &filename ) const;

    void                       add_repeated           ( const parex::String &shape_name, const parex::Scalar &count, const parex::Tensor<> &coordinates, const parex::Vector<> &face_ids, const parex::Scalar &beg_ids = 0 );
    //void                     plane_cut              ( const parex::String &normals, const Value &scalar_products, const Value &cut_ids );

private:
    parex::Rc<parex::Variable> elem_info;             ///<
    parex::Rc<parex::Variable> shape_map;             ///<
};

}
