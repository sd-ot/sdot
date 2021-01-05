#ifndef SDOT_HomogeneousElementaryPolytopList_HEADER
#define SDOT_HomogeneousElementaryPolytopList_HEADER

#include <parex/containers/tensor.h>

template<class Allocator_TF,class Allocator_TI,int nb_nodes,int nb_faces,int dim>
struct HomogeneousElementaryPolytopList {
    using                      TF                              = typename Allocator_TF::value_type;
    using                      TI                              = typename Allocator_TI::value_type;

    using                      XP                              = parex::tensor<parex::heap_tensor_block<Allocator_TF,3,parex::DynamicShapeFactory<nb_nodes,dim,parex::unspecified_size>>>;
    using                      XF                              = parex::tensor<parex::heap_tensor_block<Allocator_TI,2,parex::DynamicShapeFactory<nb_faces,parex::unspecified_size>>>;
    using                      XI                              = parex::tensor<parex::heap_tensor_block<Allocator_TI,1,parex::DynamicShapeFactory<>>>;

    /**/                       HomogeneousElementaryPolytopList( const Allocator_TF &allocator_TF, const Allocator_TI &allocator_TI, TI rese_items = 0 );

    void                       write_to_stream                 ( std::ostream &os, const std::string &sp = "\n" ) const;
    template<class Proc> void  resize                          ( TI new_size, const Proc &proc );
    TI                         size                            () const;

    XP                         positions;                      ///< ( num_node, num_dim, num_item )
    XF                         face_ids;                       ///< ( num_face, num_item )
    XI                         ids;                            ///< ( num_item )
};

#include "HomogeneousElementaryPolytopList.tcc"

#endif // SDOT_HomogeneousElementaryPolytopList_HEADER
