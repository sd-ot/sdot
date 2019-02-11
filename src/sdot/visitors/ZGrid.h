#pragma once

#include "../ConvexPolyhedron3.h"
#include "../ConvexPolyhedron2.h"
#include <functional>

namespace sdot {

/**
  Pc is expected to contain
  - static int dim
  - TF => floating point type
  - TI => index type

*/
template<class Pc>
class ZGrid {
public:
    // data from Pc
    static constexpr int    nb_bits_per_axis      = Pc::nb_bits_per_axis;
    static constexpr int    dim                   = Pc::dim;
    using                   TF                    = typename Pc::TF;
    using                   TI                    = typename Pc::TI;

    // static definitions
    static constexpr bool   full_may_cut_test     = false;
    static constexpr int    sizeof_zcoords        = ( dim * nb_bits_per_axis + 7 ) / 8; ///< nb meaningful bytes in z-coordinates
    using                   CP2                   = ConvexPolyhedron2<Pc,TI>;
    using                   CP3                   = ConvexPolyhedron3<Pc,TI>;
    using                   CP                    = typename std::conditional<dim==3,CP3,CP2>::type;
    using                   Pt                    = typename CP::Pt;
    using                   TZ                    = std::uint64_t; ///< zcoords

    // methods
    /* ctor */              ZGrid                 ( std::size_t max_diracs_per_cell = 11, TF max_delta_weight_per_grid = 1e40 );

    void                    update                ( const Pt *positions, const TF *weights, std::size_t nb_diracs, bool positions_have_changed = true, bool weights_have_changed = true, bool ball_cut = false );
    template<int bc> void   update                ( const Pt *positions, const TF *weights, std::size_t nb_diracs, bool positions_have_changed, bool weights_have_changed, N<bc> ball_cut );

    int                     for_each_laguerre_cell( const std::function<void( CP &lc, std::size_t num )> &f, const CP &starting_lc, const Pt *positions, const TF *weights, std::size_t nb_diracs, bool stop_if_void_lc = false ); ///< starting_lc can be a polygonal bound
    int                     for_each_laguerre_cell( const std::function<void( CP &lc, std::size_t num, int num_thread )> &f, const CP &starting_lc, const Pt *positions, const TF *weights, std::size_t nb_diracs, bool stop_if_void_lc = false, bool ball_cut = false ); ///< version with num_thread
    template<int bc> int    for_each_laguerre_cell( const std::function<void( CP &lc, std::size_t num, int num_thread )> &f, const CP &starting_lc, const Pt *positions, const TF *weights, std::size_t nb_diracs, bool stop_if_void_lc, N<bc> ball_cut ); ///< version with num_thread

    bool                    check_sanity          ( const Pt *positions ) const;
    std::size_t             nb_levels             () const { return grids.size(); }
    template<class V> void  display               ( V &vtk_output, TF z = 0 ) const; ///< for debug purpose

    // values used by init
    TF                      max_delta_weight_per_grid;
    int                     max_diracs_per_cell;
    bool                    eq_rep_weight_split;
    std::vector<Pt>         translations;

private:
    struct                  Cell {
        TI                  dpc_offset;                 ///< offsets in grid.dpc_values
        // TF               max_weight;
        TZ                  zcoords;
        TF                  size;
        Pt                  pos;
    };

    struct                  Grid {
        std::vector<TI>     cell_index_vs_dirac_number; ///< num cell for each dirac
        std::vector<TI>     dirac_indices;              ///< filled only if blocks.size() > 1
        std::vector<TI>     dpc_values;                 ///< diracs indices for each cell
        std::vector<TI>     ng_indices;                 ///< list of cell index of direct neighbors
        std::vector<TI>     ng_offsets;                 ///< offsets in ng_indices for each cell
        TF                  min_weight;
        TF                  max_weight;
        std::vector<Cell>   cells;
    };

    struct                  ZNode {
        void                write_to_stream( std::ostream &os ) const { os << zcoords << "[" << index << "]"; }
        TZ                  operator>>     ( int shift ) const { return zcoords >> shift; }
        TZ                  zcoords; ///<
        TI                  index;
    };

    void                    fill_grid_using_zcoords( TI num_grid, const Pt *positions, const TF *weights, std::size_t nb_diracs );
    void                    repl_zcoords_by_ccoords( TI num_grid, const TF *weights );
    void                    find_englobing_cousins ( TI num_grid, const Pt *positions ); ///< find englobing cells for each dirac (and for each grid). Must be done after repl_zcoords_by_ccoords
    void                    update_the_limits      ( const Pt *positions, const TF *weights, std::size_t nb_diracs );
    void                    update_neighbors       ( TI num_grid );
    void                    fill_the_grids         ( const Pt *positions, const TF *weights, std::size_t nb_diracs );
    template<int bc> TF     min_w_to_cut           ( const CP &lc, Pt c0, TF w0, const Cell &cr_cell, const Pt *positions, const TF *weights, N<bc> );
    template<class C> TZ    zcoords_for            ( const C &pos ); ///< floating point position
    template<int d>   TZ    ng_zcoord              ( TZ zcoords, TZ off, N<d> ) const;
    //    bool              may_cut                ( const CP &lc, TI i0, const Grid &cr_grid, const Cell &cr_cell, const Pt *positions, const TF *weights );

    // true if a dirac in b1 (given max_weight in b1 and its neighbors) may cut lc

    // tmp
    using                   RsTmp                  = std::vector<std::array<std::size_t,256>>;
    RsTmp                   rs_tmps;
    std::vector<ZNode>      znodes;                ///< tmp znodes
    std::vector<ZNode>      zcells;                ///<

    //
    std::vector<TI>         num_grid_vs_weight;    ///<
    TF                      step_length;
    TF                      grid_length;
    TF                      min_weight;
    TF                      max_weight;
    Pt                      min_point;
    Pt                      max_point;
    std::vector<Grid>       grids;                 ///< for each weight span
};

} // namespace sdot

#include "ZGrid.tcc"

