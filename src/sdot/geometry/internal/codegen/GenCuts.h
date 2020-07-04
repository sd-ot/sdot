#ifndef SDOT_GEN_CUTS_HEADER
#define SDOT_GEN_CUTS_HEADER

#include "../../../support/Rational.h"
#include "../../RecursivePolytop.h"
#include "../../VtkOutput.h"
#include "GlobalGenCutData.h"
#include <deque>

template<int dim>
struct GenCuts {
    using                TF                        = Rational;
    using                TI                        = std::size_t;
    using                Pt                        = Point<TF,dim>;
    struct               CutNode                   { std::array<TI,2> inds; Pt pos; void write_to_stream( std::ostream &os ) const { os << "[" << inds << "]"; } };
    using                Rp                        = RecursivePolytop<TF,dim,TI,CutNode>;
    struct               Shape                     { Rp rp; std::string name; };
    struct               Part                      { Rp rp; const Shape *ref_shape; TF measure; std::vector<CutNode> cut_nodes; std::vector<Part *> compatible_parts; void write_to_stream( std::ostream &os ) const { os << ref_shape->name << " " << cut_nodes; } };

    /**/                 GenCuts                   ( GlobalGenCutData &global_cut_data );

    void                 add_ref_shape             ( std::string name );
    void                 setup_cut_nodes_for       ( const Shape &ref_shape );
    void                 setup_parts_from_cut_nodes();
    void                 make_best_combs_from_parts( const std::vector<Part *> &chosen_parts = {}, const std::vector<Part *> &compatible_parts = {}, TF measure = 0, const std::set<std::array<TI,2>> &used_points = {} );
    void                 makes_comb_for_cases      ();
    void                 write_code_for_cases      ();

    void                 display_ref_shape         () const;
    void                 display_parts             () const;
    void                 display_best_combs        () const;

// private:
    struct               Comb                      { std::vector<Part *> parts; TF measure; std::pair<TF,TF> score() const { return { measure, - TF( parts.size() ) }; } };
    using                CombMap                   = std::map<std::set<std::array<TI,2>>,Comb>;
    void                 write_case                ( std::ostream &os, TI num_case );

    GlobalGenCutData&    global_cut_data;
    std::deque<Shape>    ref_shapes;
    const Shape*         ref_shape_to_cut;
    std::vector<CutNode> cut_nodes;
    std::deque<Part>     parts;
    CombMap              best_combs;
    std::vector<Comb>    comb_for_cases;
};

#include "GenCuts.tcc"

#endif // SDOT_GEN_CUTS_HEADER
