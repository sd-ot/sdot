#include <parex/containers/xtensor.h>

namespace sdot {

template<class TF,class TI,int nb_nodes,int dim>
struct HomogeneousElementaryPolytopList {
    TI                size       () const { return ids.size(); }

    xt::xtensor<TF,3> positions; ///< ( num_item, num_node, num_dim )
    xt::xtensor<TI,2> face_ids;  ///< ( num_item, num_node )
    xt::xtensor<TI,1> ids;       ///< ( num_item )
};

}
