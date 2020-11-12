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
    ShapeData<TF,TI,dim> &shape_data     ( ShapeType *shape_type, TI nb_items_if_creation = 0 );
    static std::string    type_name      ();

    Map                   map;           ///<
};

} // namespace sdot

#include "ShapeMap.tcc"

#endif // SDOT_ShapeMap_H
