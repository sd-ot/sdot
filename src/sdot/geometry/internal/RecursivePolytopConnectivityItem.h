#ifndef SDOT_RecursivePolytopConnectivityItem_HEADER
#define SDOT_RecursivePolytopConnectivityItem_HEADER

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
    using                             ItemPool         = RecursivePolytopConnectivityItemPool<TF_,TI_,nvi_>;
    using                             Face             = RecursivePolytopConnectivityItem<TF_,TI_,std::max(nvi_-1,0)>;
    using                             Item             = RecursivePolytopConnectivityItem;
    enum {                            nvi              = nvi_ };
    using                             TF               = TF_;
    using                             TI               = TI_;

    template<class Pt> static void    add_convex_hull  ( ItemPool &item_pool, BumpPointerPool &mem_pool, std::vector<Item *> &res, const Pt *positions, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &normal, const Pt &center );

    void                              write_to_stream  ( std::ostream &os ) const;

    std::vector<Face *>               sorted_faces;    ///<
    RecursivePolytopConnectivityItem* prev_in_pool;    ///<
    RecursivePolytopConnectivityItem* sibling;         ///<
    mutable TI                        num;             ///<
};

/** Vertex */
template<class TF_,class TI_>
struct RecursivePolytopConnectivityItem<TF_,TI_,0> {
    using                             ItemPool         = RecursivePolytopConnectivityItemPool<TF_,TI_,0>;
    using                             Face             = Void;
    using                             Item             = RecursivePolytopConnectivityItem;
    enum {                            nvi              = 0 };
    using                             TF               = TF_;
    using                             TI               = TI_;

    template<class Pt> static void    add_convex_hull  ( ItemPool &item_pool, BumpPointerPool &mem_pool, std::vector<Item *> &res, const Pt *positions, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &normal, const Pt &center );

    void                              write_to_stream  ( std::ostream &os ) const;

    RecursivePolytopConnectivityItem* prev_in_pool;    ///<
    TI                                node_number;     ///<
    RecursivePolytopConnectivityItem* sibling;         ///<
    mutable TI                        num;             ///<
};

#include "RecursivePolytopConnectivityItem.tcc"

#endif // SDOT_RecursivePolytopConnectivityItem_HEADER
