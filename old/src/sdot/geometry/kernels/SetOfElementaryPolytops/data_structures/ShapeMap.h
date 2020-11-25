#ifndef SDOT_ShapeMap_H
#define SDOT_ShapeMap_H

#include "ElementaryPolytopOperations.h"
#include "ShapeData.h"
#include <map>

namespace sdot {

/***/
template<class TF,class TI,int dim>
struct ShapeMap {
    using                 Map            = std::map<std::string,ShapeData<TF,TI,dim>>;

    void                  write_to_stream( std::ostream &os ) const;
    ShapeData<TF,TI,dim> &shape_data     ( const std::string &shape_name, const ElementaryPolytopOperations &eto, TI nb_items_if_creation = 0 );
    static std::string    type_name      ();

    Map                   map;           ///<
};

} // namespace sdot

#include "ShapeMap.tcc"

#endif // SDOT_ShapeMap_H
