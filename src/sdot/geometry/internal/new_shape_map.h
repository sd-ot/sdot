#include <sdot/geometry/kernels/SetOfElementaryPolytops/data_structures/ShapeMap.h>
#include <parex/support/S.h>
#include <parex/support/N.h>
#include <map>

using namespace parex;
using namespace sdot;

template<class TF,class TI,int dim>
ShapeMap<TF,TI,dim> *new_shape_map( S<TF>, S<TI>, N<dim> ) {
    return new ShapeMap<TF,TI,dim>;
}
