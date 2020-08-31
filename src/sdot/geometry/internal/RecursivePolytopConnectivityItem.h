#ifndef SDOT_RecursivePolytopConnectivityItem_HEADER
#define SDOT_RecursivePolytopConnectivityItem_HEADER

#include "../../support/Void.h"
#include "../Point.h"
#include <vector>

/**
*/
template<class TF_,class TI_,int nvi_>
struct RecursivePolytopConnectivityItem {
    using                             TF               = TF_;
    using                             TI               = TI_;
    enum {                            nvi              = nvi_ };
    using                             Face             = RecursivePolytopConnectivityItem<TF,TI,std::max(nvi-1,0)>;
    struct                            OrientedFace     { Face *ref; bool neg; };

    void                              write_to_stream  ( std::ostream &os ) const;

    RecursivePolytopConnectivityItem *prev_in_pool;    ///<
    std::vector<OrientedFace>         faces;           ///<
};

/** Edge */
template<class TF_,class TI_>
struct RecursivePolytopConnectivityItem<TF_,TI_,0> {
    using                             TF               = TF_;
    using                             TI               = TI_;
    enum {                            nvi              = 1 };
    using                             Face             = Void;

    void                              write_to_stream  ( std::ostream &os ) const;

    RecursivePolytopConnectivityItem *prev_in_pool;    ///<
    TI                                node_number;     ///<
};

#include "RecursivePolytopConnectivityItem.tcc"

#endif // SDOT_RecursivePolytopConnectivityItem_HEADER
