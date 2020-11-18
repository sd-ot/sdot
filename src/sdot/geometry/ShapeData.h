#ifndef SDOT_ShapeData_H
#define SDOT_ShapeData_H

#include <parex/containers/Tensor.h>
#include <parex/containers/Vec.h>

namespace sdot {
class ShapeType;

/***/
template<class TF,class TI,int dim>
struct ShapeData {
    /**/               ShapeData      ( TI nb_nodes, TI nb_faces, TI rese = 0, TI size = 0 );

    void               write_to_stream( std::ostream &os, const std::string &sp = "  " ) const;
    static std::string type_name()    { return "sdot::ShapeData<" + parex::type_name<TF>() + "," + parex::type_name<TI>() + "," + std::to_string( dim ) + ">"; }

    parex::Tensor<TF>  coordinates;   ///< all the x for node 0, all the y for node 0, ... all the x for node 1, ...
    parex::Tensor<TI>  face_ids;      ///< all the ids for node 0, all the ids for node 1, ...
    TI                 size;          ///<
    parex::Vec<TI>     ids;           ///<
};

} // namespace sdot

#include "ShapeData.tcc"

#endif // SDOT_ShapeData_H
