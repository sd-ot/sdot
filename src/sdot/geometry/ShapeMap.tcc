#include <parex/support/ASSERT.h>
#include "ShapeMap.h"

namespace sdot {

template<class TF,class TI,int dim>
void ShapeMap<TF,TI,dim>::write_to_stream( std::ostream &os ) const {
    for( const auto &p : map )
        os << "\n " << p.first->name() << p.second;
}

template<class TF,class TI,int dim>
std::string ShapeMap<TF,TI,dim>::type_name() {
    return "sdot::ShapeMap<" + parex::type_name<TF>() + "," + parex::type_name<TI>() + "," + std::to_string( dim ) + ">";
}

template<class TF,class TI,int dim>
ShapeData<TF,TI,dim> &ShapeMap<TF,TI,dim>::shape_data( const std::string &shape_name, const ElementaryPolytopOperations &eto, TI nb_items_if_creation ) {
    auto iter = map.find( shape_name );
    if ( iter == map.end() ) {
        auto feo = eto.operation_map.find( shape_name );
        ASSERT( feo != eto.operation_map.end(), "'{}' is not a registered element type" , shape_name );

        const ElementaryPolytopOperations::Operations &eo = feo->second;
        iter = map.insert( iter, { shape_name, { eo.nb_nodes, eo.nb_faces, nb_items_if_creation } } );
    }
    return iter->second;
}

} // namespace sdot
