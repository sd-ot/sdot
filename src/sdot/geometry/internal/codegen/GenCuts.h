#ifndef SDOT_GEN_CUTS_HEADER
#define SDOT_GEN_CUTS_HEADER

#include "../../../support/Rational.h"
#include "../../RecursivePolytop.h"
#include "GlobalGenCutData.h"
#include <deque>

template<int dim>
struct GenCuts {
    using                TF                        = Rational;
    using                TI                        = std::size_t;
    using                Pt                        = Point<TF,dim>;
    using                RefRp                     = RecursivePolytop<TF,dim>;
    using                CutNodeData               = std::array<TI,2>;
    using                CutRp                     = RecursivePolytop<TF,dim,dim,TI,CutNodeData>;
    using                CutNode                   = typename CutRp::Node;
    struct               Part                      { CutRp shape;  };

    /**/                 GenCuts                   ( GlobalGenCutData &gcd );

    void                 add_ref_shape             ( std::string name );

    void                 setup_cut_nodes_for       ( const RefRp &ref_rp );
    void                 setup_parts_from_cut_nodes( const std::vector<CutNode> &choices, const std::vector<CutNode> &possibilities, const RefRp &ref_shape );
    void                 setup_parts_from_cut_nodes();

    GlobalGenCutData&    gcd;
    std::deque<RefRp>    ref_shapes;
    std::vector<CutNode> cut_nodes;
    std::deque<CutRp>    parts;
};

#include "GenCuts.tcc"

#endif // SDOT_GEN_CUTS_HEADER
