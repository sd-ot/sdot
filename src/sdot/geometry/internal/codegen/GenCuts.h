#ifndef SDOT_GEN_CUTS_HEADER
#define SDOT_GEN_CUTS_HEADER

#include "../../../support/Rational.h"
#include "../../RecursivePolytop.h"
#include "GlobalGenCutData.h"
#include <deque>

template<int dim>
struct GenCuts {
    using                TF                 = Rational;
    using                TI                 = std::size_t;
    using                Pt                 = Point<TF,dim>;
    using                RefRp              = RecursivePolytop<TF,dim>;

    struct               CutNode            { std::array<TI,2> orig_nums; };
    using                CutRp              = RecursivePolytop<TF,dim,dim,TI,CutNode>;

    /**/                 GenCuts            ( GlobalGenCutData &gcd );

    void                 add_ref_shape      ( std::string name );

    void                 setup_cut_nodes_for( RefRp *ref_rp );

    GlobalGenCutData&    gcd;
    std::deque<RefRp>    ref_shapes;
    std::vector<CutNode> cut_nodes;
    std::vector<CutRp>   parts;
};

#include "GenCuts.tcc"

#endif // SDOT_GEN_CUTS_HEADER
