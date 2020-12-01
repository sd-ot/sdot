#ifndef SDOT_HomogeneousElementaryPolytopList_HEADER
#define SDOT_HomogeneousElementaryPolytopList_HEADER

#include <parex/containers/xtensor.h>

template<class TF,class TI>
struct HomogeneousElementaryPolytopList {
    /**/              HomogeneousElementaryPolytopList( TI nb_nodes, TI nb_faces, TI dim, TI rese_items = 0 );

    void              write_to_stream                 ( std::ostream &os, const std::string &sp = "\n" ) const;
    TI                nb_nodes                        () const { return positions.shape()[ 0 ]; }
    TI                nb_faces                        () const { return face_ids.shape()[ 0 ]; }
    TI                size                            () const { return ids.size(); }
    TI                dim                             () const { return positions.shape()[ 1 ]; }

    void              resize                          ( TI new_size );

    xt::xtensor<TF,3> positions;                      ///< ( num_node, num_dim, num_item )
    xt::xtensor<TI,2> face_ids;                       ///< ( num_face, num_item )
    xt::xtensor<TI,1> ids;                            ///< ( num_item )
};

#include "HomogeneousElementaryPolytopList.tcc"

#endif // SDOT_HomogeneousElementaryPolytopList_HEADER
