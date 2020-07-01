#pragma once

#include "../ConvexPolyhedron/ConvexPolyhedron2.h"
#include "ZGridDiracSetFactory.h"
#include "ZGridUpdateParms.h"

#define ConvexPolyhedron SDOT_CONCAT_TOKEN_4( ConvexPolyhedron, DIM, _, PROFILE )
#define ZGrid SDOT_CONCAT_TOKEN_4( ZGrid_, DIM, _, PROFILE )
namespace sdot {
class VtkOutput;

/**
  Pb: on voudrait stocker d'autres trucs dans les diracs.

*/
class ZGrid {
public:
    static constexpr ST          dim                   = DIM;
    static constexpr ST          w_bounds_order        = 0;
    using                        Pt                    = Point<TF,dim>;
    using                        UpdateParms           = ZGridUpdateParms<TF,ST,dim>;
    using                        CbConstruct           = std::function<void( std::array<const TF *,DIM> coords, const TF *weights, const ST *ids, ST nb_diracs )>; ///<
    struct                       TraversalFlags        { bool stop_if_void_lc = false, mod_weights = false; };

    /**/                         ZGrid                 ( ZGridDiracSetFactory<TF,ST> *dirac_set_factory = nullptr );
    /**/                        ~ZGrid                 ();

    void                         write_to_stream       ( std::ostream &os ) const;

    void                         for_each_laguerre_cell( const std::function<void( ConvexPolyhedron &lc, Dirac &dirac, int num_thread )> &f, const ConvexPolyhedron &starting_lc, TraversalFlags traversal_flags = {} );
    void                         update                ( const std::function<void( const CbConstruct & )> &f, const UpdateParms &update_parms = {} );

    ST                           max_nb_diracs_per_cell;  ///<
    ST                           available_memory;     ///<

private:
    using                        SI                    = std::int64_t;
    using                        TZ                    = std::uint64_t; ///< zcoords
    static constexpr ST          nb_coeffs_w_bound     = 1 + dim * ( w_bounds_order >= 1 ) + dim * ( dim + 1 ) / 2 * ( w_bounds_order >= 2 );
    static constexpr TZ          nb_bits_per_axis      = 20;
    static constexpr TZ          sizeof_zcoords        = ( dim * nb_bits_per_axis + 7 ) / 8; ///< nb meaningful bytes in z-coordinates
    static constexpr TZ          max_zcoords           = TZ( 1 ) << dim * nb_bits_per_axis; ///<

    struct                       Box                   { ZGridDiracSet<TF,ST> *dirac_set = nullptr; Box *sub_boxes[ 2 << dim ]; ST nb_sub_boxes = 0; Pt min_point, max_point; TZ beg_zcoord, end_zcoord; TF w_bound[ nb_coeffs_w_bound ]; };
    using                        VecBoxPtr             = std::vector<Box *>;

    void                         write_box_to_stream   ( std::ostream &os, const Box *box, std::string sp ) const;
    void                         get_min_and_max_pts   ( const std::function<void( const CbConstruct & )> &f, const UpdateParms &update_parms, ST &nb_diracs );
    void                         make_the_boxes_rec    ( Box **boxes, ST &nb_boxes, const std::vector<SI> &h, ST beg_h, ST end_h, ST off_h, ST mul_h );
    void                         update_histogram      ( const std::function<void( const CbConstruct & )> &f, const UpdateParms &update_parms, ST approx_nb_diracs );
    void                         make_the_boxes        ( const std::function<void( const CbConstruct & )> &f, const UpdateParms &update_parms );
    void                         fill_the_boxes        ( const std::function<void( const CbConstruct & )> &f, const UpdateParms &update_parms );
    void                         for_each_box          ( const std::function<void( Box *box )> &f, Box **boxes, ST nb_boxes );

    ZGridDiracSetFactory<TF,ST>* dirac_set_factory;
    TF                           inv_step_length;
    TF                           step_length;
    TF                           grid_length;
    std::vector<VecBoxPtr>       final_boxes;          ///<
    std::vector<std::vector<SI>> histograms;           ///< < 0 => next level of hist
    Pt                           min_point;
    Pt                           max_point;
    ST                           nb_diracs;

    BumpPointerPool              box_pool;
    ST                           nb_boxes;
    Box*                         boxes[ 2 << dim ]; ///<
};

} // namespace sdot
#undef ZGrid