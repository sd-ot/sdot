#pragma once

#include "../ConvexPolyhedron/ConvexPolyhedron2.h"
#include "ZGridDiracSetFactory.h"
#include "ZGridUpdateParms.h"

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
    static constexpr ST          dim               = DIM;
    using                        Pt                = Point<TF,dim>;
    using                        UpdateParms       = ZGridUpdateParms<TF,ST,dim>;
    using                        CbConstruct       = std::function<void( std::array<const TF *,DIM> coords, const TF *weights, const ST *ids, ST nb_diracs )>; ///<

    /**/                         ZGrid             ( ZGridDiracSetFactory<TF,ST> *dirac_set_factory = nullptr );
    /**/                        ~ZGrid             ();

    void                         update            ( const std::function<void( const CbConstruct & )> &f, const UpdateParms &update_parms = {} );

    ST                           available_memory; ///<

private:
    using                        TZ                = std::uint64_t; ///< zcoords
    static constexpr TZ          nb_bits_per_axis  = 20;
    static constexpr TZ          sizeof_zcoords    = ( dim * nb_bits_per_axis + 7 ) / 8; ///< nb meaningful bytes in z-coordinates
    static constexpr TZ          max_zcoords       = TZ( 1 ) << dim * nb_bits_per_axis; ///<

    void                         get_dimensions    ( const std::function<void( const CbConstruct & )> &f, const UpdateParms &update_parms );
    void                         make_the_cells    ( const std::function<void( const CbConstruct & )> &f, const UpdateParms &update_parms );

    ZGridDiracSetFactory<TF,ST>* dirac_set_factory;
    TF                           inv_step_length;
    TF                           step_length;
    TF                           grid_length;
    Pt                           min_point;
    Pt                           max_point;
    ST                           nb_diracs;
    std::vector<std::vector<ST>> hist;             ///<
};

} // namespace sdot
#undef ZGrid
