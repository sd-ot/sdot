#ifndef SDOT_RecursivePolytopConnectivityItem_HEADER
#define SDOT_RecursivePolytopConnectivityItem_HEADER

#include "RecursivePolytopConnectivityFace.h"
#include "../../support/BumpPointerPool.h"
#include "../../support/Void.h"
#include "../Point.h"
#include <vector>

template<class TF,class TI,int nvi>
struct RecursivePolytopConnectivityItemPool;

/**
*/
template<class TF_,class TI_,int nvi_>
struct RecursivePolytopConnectivityItem {
    using                             FaceItem         = RecursivePolytopConnectivityItem<TF_,TI_,std::max(nvi_-1,0)>;
    using                             ItemPool         = RecursivePolytopConnectivityItemPool<TF_,TI_,nvi_>;
    using                             Vertex           = RecursivePolytopConnectivityItem<TF_,TI_,0>;
    using                             Face             = RecursivePolytopConnectivityFace<FaceItem>;
    using                             Item             = RecursivePolytopConnectivityItem;
    using                             VVPI             = std::vector<std::vector<Item *>>;
    enum {                            nvi              = nvi_ };
    using                             TF               = TF_;
    using                             TI               = TI_;

    template<class Pt> static void    add_convex_hull  ( std::vector<Item *> &res, ItemPool &item_pool, BumpPointerPool &mem_pool, const Pt *positions, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &center );

    void                              write_to_stream  ( std::ostream &os ) const;
    const Vertex*                     first_vertex     () const { return faces[ 0 ].item->first_vertex(); }
    Vertex*                           first_vertex     () { return faces[ 0 ].item->first_vertex(); }
    template<class Pt> Item*          copy_rec         ( std::vector<Pt> &new_positions, ItemPool &new_item_pool, BumpPointerPool &new_mem_pool, const std::vector<Pt> &old_positions ) const;

    RecursivePolytopConnectivityItem* prev_in_pool;    ///<
    std::vector<Face>                 faces;           ///<

    mutable VVPI                      new_items;       ///< list of possibilities, each one potentially containing several sub-items
    mutable Item*                     new_item;        ///< used for copies
    mutable TI                        tmp_num;             ///<
};

/** Vertex */
template<class TF_,class TI_>
struct RecursivePolytopConnectivityItem<TF_,TI_,0> {
    using                             ItemPool         = RecursivePolytopConnectivityItemPool<TF_,TI_,0>;
    using                             FaceItem         = Void;
    using                             Face             = Void;
    using                             Item             = RecursivePolytopConnectivityItem;
    using                             VVPI             = std::vector<std::vector<Item *>>;
    enum {                            nvi              = 0 };
    using                             TF               = TF_;
    using                             TI               = TI_;

    template<class Pt> static void    add_convex_hull  ( std::vector<Item *> &res, ItemPool &item_pool, BumpPointerPool &mem_pool, const Pt *positions, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &center );

    void                              write_to_stream  ( std::ostream &os ) const;
    const Item*                       first_vertex     () const { return this; }
    Item*                             first_vertex     () { return this; }
    template<class Pt> Item*          copy_rec         ( std::vector<Pt> &new_positions, ItemPool &new_item_pool, BumpPointerPool &new_mem_pool, const std::vector<Pt> &old_positions ) const;

    RecursivePolytopConnectivityItem* prev_in_pool;    ///<
    TI                                node_number;     ///<

    mutable VVPI                      new_items;       ///< list of possibilities, each one potentially containing several sub-items
    mutable Item*                     new_item;        ///< used for copies
    mutable TI                        tmp_num;         ///<
    mutable TF                        sp;              ///< scalar product
};

#include "RecursivePolytopConnectivityItem.tcc"

#endif // SDOT_RecursivePolytopConnectivityItem_HEADER
