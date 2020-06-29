#ifndef SDOT_CONVEX_HULL_INDICES_H
#define SDOT_CONVEX_HULL_INDICES_H

#include "../../../support/VtkOutput.h"
#include "../../../support/Rational.h"
#include <functional>
#include <vector>
#include <set>

/**/
template<int dim,class TF=Rational>
class ConvexHullIndices;

/**/
template<class TF_>
class ConvexHullIndices<1,TF_> {
public:
    using                   TI                 = std::size_t;
    using                   TF                 = TF_;
    using                   Pt                 = Point<TF,1>;
    using                   Sizes              = TI;

    template<class Pu>      ConvexHullIndices  ( const std::vector<Pt> &pts, const std::vector<TI> &inds, const std::vector<Pu> &ori_pts, const std::vector<Pu> &ori_dirs );
    /**/                    ConvexHullIndices  ( const std::vector<Pt> &pts = {} );

    template<class Pu> void adjust_orientation ( const std::vector<Pu> &ori_pts, const std::vector<Pu> &ori_dirs = {} );
    void                    replace_inds       ( const TI *repl );

    void                    write_to_stream    ( std::ostream &os, std::string sp = "" ) const;
    void                    get_links_rec      ( std::vector<std::set<TI>> &links ) const;
    Sizes                   sorted_sizes       () const { return inds.size(); }
    template<class Pu> void display_vtk        ( VtkOutput &vo, const std::vector<Pu> &pts ) const;
    template<class Pu> void on_normal          ( const std::vector<Pu> &pts, const std::function<void( Pu ori, Pu dir )> &f, std::vector<Pu> dirs = {} ) const;
    bool                    operator<          ( const ConvexHullIndices &that ) const;
    template<class Pu> TF   measure            ( const std::vector<Pu> &ori_pts, const std::vector<Pu> &ori_dirs = {}, TF div = 1 ) const;
    template<class Pu> Pu   center             ( const std::vector<Pu> &ori_pts ) const;

    std::vector<TI>         inds;
};

/**/
template<int dim,class TF_>
class ConvexHullIndices {
public:
    using                   TI                 = std::size_t;
    using                   TF                 = TF_;
    using                   Pt                 = Point<TF,dim>;
    using                   Next               = ConvexHullIndices<dim-1,TF>;
    using                   Sizes              = std::vector<typename Next::Sizes>;

    template<class Pu>      ConvexHullIndices  ( const std::vector<Pt> &loc_pts, const std::vector<TI> &loc_inds, const std::vector<Pu> &ori_pts, const std::vector<Pu> &ori_dirs );
    /**/                    ConvexHullIndices  ( const std::vector<Pt> &pts = {} );

    void                    replace_inds       ( const TI *repl );

    bool                    is_a_permutation_of( const ConvexHullIndices &that, TI *perm_this_to_that ) const;
    void                    write_to_stream    ( std::ostream &os, std::string sp = "" ) const;
    void                    get_links_rec      ( std::vector<std::set<TI>> &links ) const;
    Sizes                   sorted_sizes       () const;
    template<class Pu> void display_vtk        ( VtkOutput &vo, const std::vector<Pu> &pts ) const;
    template<class Pu> void on_normal          ( const std::vector<Pu> &pts, const std::function<void( Pu ori, Pu dir )> &f, std::vector<Pu> dirs = {} ) const;
    bool                    operator<          ( const ConvexHullIndices &that ) const;
    template<class Pu> TF   measure            ( const std::vector<Pu> &ori_pts, const std::vector<Pu> &ori_dirs = {}, TF div = dim ) const;
    template<class Pu> Pu   center             ( const std::vector<Pu> &ori_pts ) const;

    std::vector<Next>       nexts;

private:
    using                   VTI                = std::vector<TI>;

    bool                    test_permutations  ( const ConvexHullIndices &that, TI *perm, const std::vector<std::set<TI>> &this_links, const std::vector<std::set<TI>> &that_links, const std::vector<std::set<TI>> &possibilities ) const;
    template<int d> VTI     ordered_pt_seq     ( N<d> ) const { return {}; }
    VTI                     ordered_pt_seq     ( N<2> ) const;
};

#include "ConvexHullIndices.tcc"

#endif // SDOT_CONVEX_HULL_INDICES_H

