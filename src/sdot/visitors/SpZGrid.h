#pragma once

#include "../ConvexPolyhedron2.h"
#include "../ConvexPolyhedron3.h"
#include <functional>

namespace sdot {

/**
  Pc is expected to contain
  - static int dim
  - TF => floating point type
  - TI => index type

*/
template<class Pc>
class SpZGrid {
public:
    // data from Pc
    static constexpr int    dim                   = Pc::dim;
    using                   TF                    = typename Pc::TF;
    using                   TI                    = typename Pc::TI;

    // parameters
    static constexpr bool   allow_translations    = true;
    static constexpr int    degree_w_approx       = 1;
    static constexpr bool   allow_mpi             = true;

    // static definitions
    static constexpr int    nb_coeffs_w_approx    = 1 + dim * ( degree_w_approx >= 1 ) + dim * ( dim + 1 ) / 2 * ( degree_w_approx >= 2 );
    using                   CP2                   = ConvexPolyhedron2<Pc,TI>;
    using                   CP3                   = ConvexPolyhedron3<Pc,TI>;
    using                   CP                    = typename std::conditional<dim==3,CP3,CP2>::type;
    using                   Pt                    = typename CP::Pt;

    // methods
    /* ctor */              SpZGrid               ( TI max_diracs_per_cell = 11 );

    void                    update                ( const Pt *positions, const TF *weights, TI nb_diracs, bool positions_have_changed = true, bool weights_have_changed = true );

    int                     for_each_laguerre_cell( const std::function<void( CP &lc, TI num )> &f, const CP &starting_lc, const Pt *positions, const TF *weights, TI nb_diracs, bool stop_if_void_lc = false ); ///< starting_lc can be a polygonal bound
    int                     for_each_laguerre_cell( const std::function<void( CP &lc, TI num, int num_thread )> &f, const CP &starting_lc, const Pt *positions, const TF *weights, TI nb_diracs, bool stop_if_void_lc = false, bool ball_cut = false ); ///< version with num_thread
    template<int bc> int    for_each_laguerre_cell( const std::function<void( CP &lc, TI num, int num_thread )> &f, const CP &starting_lc, const Pt *positions, const TF *weights, TI nb_diracs, bool stop_if_void_lc, N<bc> ball_cut ); ///< version with num_thread

    template<class V> void  display               ( V &vtk_output, TF z = 0 ) const; ///< for debug purpose

    // values used by update
    int                     max_diracs_per_cell;
    int                     depth_initial_send;
    std::vector<Pt>         translations;         ///< symetries

private:
    using                   CoeffsWApprox         = std::array<TF,nb_coeffs_w_approx>;

    struct                  PWI {
        TI                  num_dirac;
        Pt                  position;
        TF                  weight;
    };

    struct                  Box {
        float               dist_2                ( Pt p, TF w ) const;

        CoeffsWApprox       coeffs_w_approx;
        TI                  beg_indices;          ///< only for local boxes
        TI                  end_indices;          ///< only for local boxes
        Box*                last_child;
        Box*                sibling;
        std::vector<PWI>    ext_pwi;              ///< only for external boxes
        Pt                  min_pt;
        Pt                  max_pt;
        TI                  depth;
        int                 rank;
    };
    struct                  Neighbor {
        std::size_t         mpi_rank;
        Box*                root;
    };

    Box*                    deserialize_rec       ( const std::vector<char> &dst, int ext_rank );
    std::vector<char>       serialize_rec         ( const Pt *positions, const TF *weights, std::vector<Box *> front, int max_depth );
    void                    initial_send          ( const Pt *positions, const TF *weights );
    void                    update_box            ( const Pt *positions, const TF *weights, Box *box, TI beg_indices, TI end_indices, TI depth );

    std::vector<TI>         dirac_indices;
    std::vector<Neighbor>   neighbors;
    std::deque<Box>         boxes;
    Box*                    root;
};

} // namespace sdot

#include "SpZGrid.tcc"

