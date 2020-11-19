#ifndef SDOT_ShapeMap_H
#define SDOT_ShapeMap_H

#include "ShapeData.h"
#include "ShapeType.h"
#include <map>

namespace sdot {

/***/
template<class TF,class TI,int dim>
struct ShapeMap {
    using                 Map            = std::map<ShapeType *,ShapeData<TF,TI,dim>>;

    void                  write_to_stream( std::ostream &os ) const;
    static std::string    type_name      ();

    ShapeData<TF,TI,dim> &operator[]     ( ShapeType *shape_type );

    Map                   map;           ///<
};

} // namespace sdot

#include "ShapeMap.tcc"

#endif // SDOT_ShapeMap_H
