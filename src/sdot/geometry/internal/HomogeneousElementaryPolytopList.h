#ifndef SDOT_HomogeneousElementaryPolytopList_HEADER
#define SDOT_HomogeneousElementaryPolytopList_HEADER

#include <parex/containers/gtensor.h>

template<class Allocator_TF,class Allocator_TI>
struct HomogeneousElementaryPolytopList {
    using TF                              = typename Allocator_TF::value_type;
    using TI                              = typename Allocator_TI::value_type;

    using XF3                             = gtensor<TF,3,Allocator_TF>;
    using XI2                             = gtensor<TI,2,Allocator_TI>;
    using XI1                             = gtensor<TI,1,Allocator_TI>;

    /**/  HomogeneousElementaryPolytopList( Allocator_TF &allocator_TF, Allocator_TI &allocator_TI, TI nb_nodes, TI nb_faces, TI dim, TI rese_items = 0 );

    void  write_to_stream                 ( std::ostream &os, const std::string &sp = "\n" ) const;
    TI    nb_nodes                        () const { return positions.shape()[ 0 ]; }
    TI    nb_faces                        () const { return face_ids.shape()[ 0 ]; }
    TI    size                            () const { return ids.size(); }
    TI    dim                             () const { return positions.shape()[ 1 ]; }

    void  resize                          ( Allocator_TF &allocator_TF, Allocator_TI &allocator_TI, TI new_size );

    XF3   positions;                      ///< ( num_node, num_dim, num_item )
    XI2   face_ids;                       ///< ( num_face, num_item )
    XI1   ids;                            ///< ( num_item )
};

#include "HomogeneousElementaryPolytopList.tcc"

#endif // SDOT_HomogeneousElementaryPolytopList_HEADER
