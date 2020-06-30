#ifndef SDOT_CONVEX_HULL_H
#define SDOT_CONVEX_HULL_H

#include "ConvexHullIndices.h"

/**/
template<int dim,class TF_=Rational>
class ConvexHull {
public:
    using           Chi                = ConvexHullIndices<dim,TF_>;
    using           TI                 = typename Chi::TI;
    using           Pt                 = typename Chi::Pt;
    using           TF                 = TF_;

    /**/            ConvexHull         ( const std::vector<Pt> &pts = {}, std::string friendly_name = {} );

    bool            is_a_permutation_of( const ConvexHull &that, TI *perm_this_to_that ) const;
    void            write_to_stream    ( std::ostream &os ) const;
    void            for_each_normal    ( const std::function<void( Pt ori, Pt dir )> &f ) const;
    ConvexHull      intersection       ( const ConvexHull &that ) const;
    void            display_vtk        ( VtkOutput &vo, Pt offset = TF( 0 ) ) const;
    TF              measure            () const;
    Pt              center             () const;
    std::string     name               () const;
    ConvexHull      cut                ( Pt orig, Pt normal ) const;

    std::string     friendly_name;
    Chi             chi;
    std::vector<Pt> pts;
};

#include "ConvexHull.tcc"

#endif // SDOT_CONVEX_HULL_H


