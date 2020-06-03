#pragma once

#include "../support/type_config.h"
#include "../support/macros.h"
#include "../support/Edge.h"
#include <functional>
#include <vector>

#define ConvexPolyhedron2 SDOT_CONCAT_TOKEN_2( ConvexPolyhedron2_, PROFILE )
namespace sdot {
class VtkOutput;

/**
*/
class ConvexPolyhedron2 {
public:
    struct                     Boundary          { Edge<TF,2> geometry; ST cut_id; };
    struct                     Vertex            {};

    static constexpr ST        dim               = 2;
    using                      Pt                = Point<TF,dim>;

    /**/                       ConvexPolyhedron2 ();
    /**/                      ~ConvexPolyhedron2 ();

    void                       init_as_box       ( Pt min, Pt max, ST cut_id = ST( -1 ) );
    void                       plane_cut         ( const TF **cd, const TF *cs, const ST *ci, ST nb_cuts );

    void                       for_each_boundary ( const std::function<void( const Boundary & )> &f );
    void                       write_to_stream   ( std::ostream &os ) const;
    void                       display_vtk       ( VtkOutput &vo ) const;
    ST                         nb_boundaries     () const { return nodes_size; }
    ST                         nb_vertices       () const { return nodes_size; }
    ST                         nb_edges          () const { return nodes_size; }
    Pt                         vertex            ( ST i ) const { return { position_xs[ i ], position_ys[ i ] }; }

private:
    TF*                        allocate_TF_vec   ( ST size );
    ST*                        allocate_ST_vec   ( ST size );
    void                       delete_TF_vec     ( TF *vec, ST size );
    void                       delete_ST_vec     ( ST *vec, ST size );
    void                       update_t_rese     ( ST t_rese );

    ST                         nodes_size;       ///< actual number of vertices
    ST                         nodes_rese;       ///< nb reserved vertices in memory

    TF*                        position_xs;      ///< position of vertices, aligned in memory
    TF*                        position_ys;      ///< position of vertices, aligned in memory
    TF*                        normal_xs;        ///< normal of edges, aligned in memory
    TF*                        normal_ys;        ///< normal of edges, aligned in memory
    ST*                        cut_ids;          ///<

    // tmp data, for which rese = nodes_rese
    std::vector<std::uint64_t> outside_nodes;    ///<
    TF*                        distances;        ///< distances from the current cutting plane, aligned in memory

    // tmp data, for which rese = t_nodes_rese
    ST                         t_nodes_rese;     ///< nb reserved vertices (or edges) in memory

    TF*                        t_position_xs;    ///< position of vertices, aligned in memory
    TF*                        t_position_ys;    ///< position of vertices, aligned in memory
    TF*                        t_normal_xs;      ///< normal of edges, aligned in memory
    TF*                        t_normal_ys;      ///< normal of edges, aligned in memory
    ST*                        t_cut_ids;        ///<
};

} // namespace sdot
#undef ConvexPolyhedron2
