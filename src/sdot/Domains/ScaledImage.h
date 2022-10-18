#pragma once

#include "../Geometry/ConvexPolyhedron2.h"
#include "../Geometry/ConvexPolyhedron3.h"
#include <mutex>

namespace sdot {

/**
  Currently, only support constant coeffs per polyhedron

  Rk: in pybind11 wraper, we have `sizes[ d ] = img.shape( img.ndim() - 1 - d );`, meaning that point[ 0 ] -> x, point[ 1 ] -> y, ...
*/
template<class Pc>
class ScaledImage {
public:
    static constexpr int   dim                        = Pc::dim;
    using                  TF                         = typename Pc::TF;
    using                  TI                         = typename Pc::TI;

    using                  CP2                        = ConvexPolyhedron2<Pc>;
    using                  CP3                        = ConvexPolyhedron3<Pc>;
    using                  CP                         = typename std::conditional<dim==3,CP3,CP2>::type;
    using                  Pt                         = typename CP::Pt;

    /**/                   ScaledImage                ( Pt min_pt = {}, Pt max_pt = {}, const TF *data = 0, std::array<TI,dim> sizes = {}, TI nb_coeffs = 1 );

    // info
    const CP&              englobing_convex_polyhedron() const;
    template<class F> void for_each_intersection      ( CP &cp, const F &f ) const; ///< f( ConvexPolyhedron, SpaceFunction )
    template<class V> void display_boundaries         ( V &vtk_output ) const;
    template<class V> void display_coeffs             ( V &vtk_output ) const;
    Pt                     min_position               () const;
    Pt                     max_position               () const;
    TI                     nb_pixels                  () const;
    TF                     measure                    () const;

    TF                     coeff_at                   ( const Pt &pos ) const;

private:
    CP                     englobing_polyheron;
    TI                     nb_coeffs;
    Pt                     min_pt;
    Pt                     max_pt;
    std::array<TI,dim>     sizes;
    std::vector<TF>        data;
};

} // namespace sdot

#include "ScaledImage.tcc"
