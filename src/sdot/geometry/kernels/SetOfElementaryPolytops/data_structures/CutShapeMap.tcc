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
ShapeData<TF,TI,dim> &ShapeMap<TF,TI,dim>::operator[]( ShapeType *shape_type ) {
    auto iter = map.find( shape_type );
    if ( iter == map.end() )
        iter = map.insert( iter, { shape_type, { shape_type } } );
    return iter->second;
}

} // namespace sdot
