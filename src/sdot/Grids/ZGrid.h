#pragma once

#include "../ConvexPolyhedron/ConvexPolyhedron2.h"
#include "ZGridDiracSetFactory.h"

#define ZGrid SDOT_CONCAT_TOKEN_4( ZGrid_, DIM, _, PROFILE )
namespace sdot {
class VtkOutput;

/**
  Pb: on voudrait stocker d'autres trucs dans les diracs.

  Prop: on fait un

  Une solution serait
*/
class ZGrid {
public:
    static constexpr ST        dim               = DIM;
    using                      Pt                = Point<TF,dim>;
    using                      CbConstruct       = std::function<void( std::array<TF *,DIM> coords, const TF *weights, const ST *ids, ST nb_diracs, bool ptrs_survive_the_call )>; ///<

    /**/                       ZGrid             ( ZGridDiracSetFactory *dirac_set_factory = nullptr );
    /**/                      ~ZGrid             ();

    void                       update            ( const std::function<void( const CbConstruct & )> &f );

private:
    static constexpr int       nb_bits_per_axis  = 20;
    static constexpr int       sizeof_zcoords    = ( dim * nb_bits_per_axis + 7 ) / 8; ///< nb meaningful bytes in z-coordinates
    using                      TZ                = std::uint64_t; ///< zcoords

    struct                     Ptrs              { std::array<TF *,DIM> coords; const TF *weights; const ST *ids; ST nb_diracs; };

    void                       get_dims          ( const std::function<void( const CbConstruct & )> &f );

    bool                       all_ptrs_survive_the_call;
    std::vector<Ptrs>          ptrs_of_previous_call;
    ZGridDiracSetFactory*      dirac_set_factory;
    TF                         inv_step_length;
    TF                         step_length;
    TF                         grid_length;
    Pt                         min_point;
    Pt                         max_point;
    ST                         nb_diracs;
};

} // namespace sdot
#undef ZGrid
