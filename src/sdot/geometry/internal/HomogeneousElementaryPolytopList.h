#ifndef SDOT_HomogeneousElementaryPolytopList_HEADER
#define SDOT_HomogeneousElementaryPolytopList_HEADER

#include <parex/containers/xtensor.h>

template<class TF,class TI>
struct HomogeneousElementaryPolytopList {
    void              write_to_stream( std::ostream &os, const std::string &sp = "\n" ) const;
    TI                size           () const { return ids.size(); }

    xt::xtensor<TF,3> positions;     ///< ( num_item, num_node, num_dim )
    xt::xtensor<TI,2> face_ids;      ///< ( num_item, num_face )
    xt::xtensor<TI,1> ids;           ///< ( num_item )
};

#include "HomogeneousElementaryPolytopList.tcc"

#endif // SDOT_HomogeneousElementaryPolytopList_HEADER
