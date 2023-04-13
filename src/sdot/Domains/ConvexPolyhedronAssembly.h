#pragma once

#include "../Geometry/ConvexPolyhedron2.h"
#include "../Geometry/ConvexPolyhedron3.h"
#include <mutex>

namespace sdot {

/**
  Currently, only support constant coeffs per polyhedron
*/
template<class Pc>
class ConvexPolyhedronAssembly {
public:
    static constexpr int   dim                        = Pc::dim;
    using                  TF                         = typename Pc::TF;
    using                  TI                         = typename Pc::TI;
    using                  CI                         = typename Pc::CI;

    using                  CP2                        = ConvexPolyhedron2<Pc>;
    using                  CP3                        = ConvexPolyhedron3<Pc>;
    using                  CP                         = typename std::conditional<dim==3,CP3,CP2>::type;
    using                  Pt                         = typename CP::Pt;

    // modifications
    void                   add_convex_polyhedron      ( const std::vector<Pt> &positions, const std::vector<Pt> &normals, TF coeff = 1.0, TI cut_id = -1 );
    void                   add_box                    ( Pt p0, Pt p1, TF coeff = 1.0, CI cut_id = -1 );

    void                   normalize                  ();

    // info
    const CP&              englobing_convex_polyhedron() const;
    template<class F> void for_each_intersection      ( CP &cp, const F &f ) const; ///< f( ConvexPolyhedron, SpaceFunction )
    template<class V> void display_boundaries         ( V &vtk_output ) const;
    template<class V> void display_coeffs             ( V &vtk_output ) const;
    Pt                     min_position               () const;
    Pt                     max_position               () const;
    TF                     measure                    () const;

    TF                     coeff_at                   ( const Pt &pos ) const;

    //
private:
    struct                 Item                       { CP polyhedron; TF coeff; };

    mutable bool           englobing_polyheron_is_up_to_date = false;
    mutable CP             englobing_polyheron;
    mutable std::mutex     mutex;
    std::vector<Item>      items;
};

} // namespace sdot

#include "ConvexPolyhedronAssembly.tcc"
