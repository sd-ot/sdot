#pragma once

#include "../Support/PoolWithActiveAndInactiveItems.h"
#include "../Integration/SpaceFunctions/Polynomial.h"
#include "../Integration/FunctionEnum.h"
#include "../Display/VtkOutput.h"
#include "Internal/Cp3Face.h"
#include "Internal/Cp3Hole.h"
#include <functional>
#include <algorithm>

#define _USE_MATH_DEFINES
#include <math.h>

namespace sdot {

/**
  Pc must contain
  - TF = floating point type
  - TI = size type (unsigned)
  - CI = information to store at each cut
  - bool allow_ball_cut = false if not planed to cut by a ball

  Beware: if there's a need for a ball_cut, it must be done AFTER the plane cuts.
*/
template<class Pc>
class ConvexPolyhedron3 {
public:
    static constexpr bool   keep_min_max_coords       = false;
    static constexpr bool   allow_ball_cut            = Pc::allow_ball_cut;
    static constexpr int    dim                       = 3;

    using                   Node                      = Cp3Node<Pc>;
    using                   Edge                      = Cp3Edge<Pc>;
    using                   Face                      = Cp3Face<Pc>;
    using                   Hole                      = Cp3Hole<Pc>;

    using                   FaceList                  = PoolWithActiveAndInactiveItems<Face>;
    using                   HoleList                  = PoolWithActiveAndInactiveItems<Hole>;
    using                   NodeList                  = PoolWithActiveAndInactiveItems<Node>;
    using                   EdgeList                  = PoolWithInactiveItems<Edge>;

    using                   TF                        = typename Face::TF; ///< floating point type
    using                   TI                        = typename Face::TI; ///< index type
    using                   CI                        = typename Face::TI; ///< cut info
    using                   Pt                        = typename Face::Pt; ///< point

    struct                  BoundaryItem              {};

    struct                  Tetra                     { Pt p0, p1, p2, p3; };
    struct                  Box                       { Pt p0, p1; };

    /// we start from a tetrahedron that includes the sphere defined by sphere_center and sphere_radius... but this sphere is not used
    /**/                    ConvexPolyhedron3         ( const Tetra &tetra, CI cut_id = {} );
    /**/                    ConvexPolyhedron3         ( const Box &box = { { 0, 0, 0 }, { 1, 1, 1 } }, CI cut_id = {} );
    /**/                    ConvexPolyhedron3         ( const ConvexPolyhedron3 &cp ) = delete;
    /**/                    ConvexPolyhedron3         ( ConvexPolyhedron3 &&cp );

    void                    operator=                 ( const ConvexPolyhedron3 &cp );
    void                    operator=                 ( ConvexPolyhedron3 &&cp );

    // display
    void                    display_html_canvas       ( std::ostream &os, TF weight, bool ext = false ) const;
    void                    write_to_stream           ( std::ostream &os ) const;
    template<class V> void  display                   ( V &vo, const typename V::CV &cell_data = {}, bool filled = true, TF max_ratio_area_error = 1e-1, bool display_tangents = false ) const;

    //
    int                     is_a_ball                 () const { return faces.empty() ? ( sphere_radius > 0 ? 1 : -1 ) : 0; } // 0 is not a ball. -1 if void, 1 if not void
    TI                      nb_points                 () const { return nodes.size(); }

    // modifications
    void                    intersect_with            ( const ConvexPolyhedron3 &cp );
    template<int no> void   plane_cut                 ( Pt origin, Pt normal, CI cut_id, N<no> normal_is_normalized ); ///< return true if effective cut
    void                    plane_cut                 ( Pt origin, Pt normal, CI cut_id = {} ) { plane_cut( origin, normal, cut_id, N<1>() ); }
    void                    ball_cut                  ( Pt center, TF radius, CI cut_id = {} ); ///< beware: only one sphere cut is authorized, and it must be done after all the plane cuts.
    void                    clear                     ( const Tetra &tetra, CI cut_id = {} );
    void                    clear                     ( const Box &box, CI cut_id = {} );

    // computations
    //    void                    for_each_boundary_measure( const SpaceFunctions::Polynomial<TF,6> &sf, const FunctionEnum::ExpWmR2db<TF> &r, const std::function<void( TF area, CI id )> &f, TF weight = 0 ) const { TODO; }
    //    void                    for_each_boundary_measure( const SpaceFunctions::Polynomial<TF,6> &sf, const FunctionEnum::Arfd          &r, const std::function<void( TF area, CI id )> &f, TF weight = 0 ) const { TODO; }
    //    void                    for_each_boundary_measure( const SpaceFunctions::Polynomial<TF,6> &sf, const FunctionEnum::WmR2          &r, const std::function<void( TF area, CI id )> &f, TF weight = 0 ) const { TODO; }
    //    void                    for_each_boundary_measure( const SpaceFunctions::Polynomial<TF,6> &sf, const FunctionEnum::Unit          &r, const std::function<void( TF area, CI id )> &f, TF weight = 0 ) const { TODO; }
    //    void                    for_each_boundary_measure( const SpaceFunctions::Polynomial<TF,6> &sf, const FunctionEnum::R2            &r, const std::function<void( TF area, CI id )> &f, TF weight = 0 ) const { TODO; }

    void                    for_each_boundary_measure ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::ExpWmR2db<TF> &r, const std::function<void( TF area, CI id )> &f, TF weight = 0 ) const;
    void                    for_each_boundary_measure ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::Arfd          &r, const std::function<void( TF area, CI id )> &f, TF weight = 0 ) const { TODO; }
    void                    for_each_boundary_measure ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::WmR2          &r, const std::function<void( TF area, CI id )> &f, TF weight = 0 ) const;
    void                    for_each_boundary_measure ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::Unit          &r, const std::function<void( TF area, CI id )> &f, TF weight = 0 ) const;
    void                    for_each_boundary_measure ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::R2            &r, const std::function<void( TF area, CI id )> &f, TF weight = 0 ) const;

    template<class F> void  for_each_boundary_item    ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::ExpWmR2db<TF> &rf, const F &f, TF weight = 0 ) const;
    template<class F> void  for_each_boundary_item    ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::Arfd          &rf, const F &f, TF weight = 0 ) const { TODO; }
    template<class F> void  for_each_boundary_item    ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::WmR2          &rf, const F &f, TF weight = 0 ) const;
    template<class F> void  for_each_boundary_item    ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::Unit          &rf, const F &f, TF weight = 0 ) const;
    template<class F> void  for_each_boundary_item    ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::R2            &rf, const F &f, TF weight = 0 ) const;

    template<class F> Node *find_node_maximizing      ( const F &f, bool return_node_only_if_true = true ) const; ///< f must return true to stop the search. It takes ( TF &value, Pt pos ) as parameters
    void                    for_each_node             ( const std::function<void( Pt v )> &f ) const;

    TF                      boundary_measure          ( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::Unit &rf ) const;
    Pt                      centroid                  ( const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::Unit &rf ) const;

    void                    add_centroid_contrib      ( Pt &ctd, TF &vol, const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::ExpWmR2db<TF> &r, TF weight = 0 ) const;
    void                    add_centroid_contrib      ( Pt &ctd, TF &vol, const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::Arfd          &r, TF weight = 0 ) const { TODO; }
    void                    add_centroid_contrib      ( Pt &ctd, TF &vol, const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::WmR2          &r, TF weight = 0 ) const;
    void                    add_centroid_contrib      ( Pt &ctd, TF &vol, const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::Unit          &r, TF weight = 0 ) const;
    void                    add_centroid_contrib      ( Pt &ctd, TF &vol, const SpaceFunctions::Constant<TF> &sf, const FunctionEnum::R2            &r, TF weight = 0 ) const;

    Pt                      random_point              () const;

    TF                      integration               ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::ExpWmR2db<TF> &r, TF weight = 0 ) const;
    TF                      integration               ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::Arfd          &r, TF weight = 0 ) const { TODO; return 0; }
    TF                      integration               ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::WmR2          &r, TF weight = 0 ) const;
    TF                      integration               ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::Unit          &r, TF weight = 0 ) const;
    TF                      integration               ( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::R2            &r, TF weight = 0 ) const;

    template<class F> bool  all_pos                   ( const F &f ) const;

    TF                      integration_der_wrt_weight( const SpaceFunctions::Constant<TF>     &sf, const FunctionEnum::ExpWmR2db<TF> &r, TF weight ) const;
    template<class FU> TF   integration_der_wrt_weight( const SpaceFunctions::Constant<TF>     &sf, const FU &r, TF weight ) const;

    void                    add_centroid_contrib     ( Pt &ctd, TF &vol ) const { return add_centroid_contrib( ctd, vol, SpaceFunctions::Constant<TF>{ 1.0 }, FunctionEnum::Unit() ); }
    TF                      boundary_measure         ()                   const { return boundary_measure( SpaceFunctions::Constant<TF>{ 1.0 }, FunctionEnum::Unit() ); }
    Pt                      centroid                 ()                   const { return centroid( SpaceFunctions::Constant<TF>{ 1.0 }, FunctionEnum::Unit() ); }
    TF                      measure                  ()                   const { return integration( SpaceFunctions::Constant<TF>{ 1.0 }, FunctionEnum::Unit() ); }

    // tests
    Pt                      min_position             () const;
    Pt                      max_position             () const;
    bool                    contains                 ( const Pt &pos ) const;
    TF                      distance                 ( const Pt &pos, bool count_domain_boundaries = false ) const; // signed distance to the boundaries (<0 means inside)
    bool                    empty                    () const { return faces.empty() && ( allow_ball_cut == false || sphere_radius <= 0 ); }
    static TF               pi                       () { if ( std::is_same<TF,double>::value ) return M_PI; using std::atan; return 4 * atan( TF( 1 ) ); }

    // approximate computations
    template<class Fu> TF   boundary_measure_ap      ( const Fu &fu, TF max_ratio_area_error = 1e-4 ) const; ///< area from a triangulation of the surface
    template<class Fu> Pt   centroid_ap              ( const Fu &fu, TI n = 1e8 ) const;                     ///< centroid, computed with monte-carlo
    template<class Fu> TF   measure_ap               ( const Fu &fu, TI n = 1e8 ) const;                     ///< volume, computed with monte-carlo

    TF                      boundary_measure_ap      ( TF max_ratio_area_error = 1e-4 ) const { return boundary_measure_ap( FunctionEnum::Unit(), max_ratio_area_error ); } ///< area from a triangulation of the surface
    Pt                      centroid_ap              ( TI n = 1e8 )                     const { return centroid_ap        ( FunctionEnum::Unit(), n                    ); } ///< centroid, computed with monte-carlo
    TF                      measure_ap               ( TI n = 1e8 )                     const { return measure_ap         ( FunctionEnum::Unit(), n                    ); } ///< volume, computed with monte-carlo

    TI                      nb_connections;          ///<
    CI                      sphere_cut_id;
    Pt                      sphere_center;
    TF                      sphere_radius;
    Pt                      min_coord;
    Pt                      max_coord;
    mutable TI              op_count;
    FaceList                faces;
    HoleList                holes;
    EdgeList                edges;
    NodeList                nodes;

private:
    struct                  EdgePair                 { Edge *a, *b; };

    struct                  MarkCutInfo              {
        struct              Gnic                     { template<class T> T *&operator()( T *e ) const { return e->next_in_cut; } };

        TI                  mod_bounds;
        ListRef<Face,Gnic>  cut_faces;
        ListRef<Edge,Gnic>  cut_edges;
        ListRef<Edge,Gnic>  rem_edges;
        ListRef<Node,Gnic>  rem_nodes;
        Pt                  origin;
        Pt                  normal;
    };

    // internal modifications methods
    void                    update_min_max_coord     ();
    EdgePair                add_straight_edge        ( Node *n0, Node *n1 );
    TI                      add_round_edge           ( Node *n0, Node *n1 );
    Node                   *add_node                 ( Pt pos );

    template<class Triangle>
    static void             p_cut                    ( std::vector<Triangle> &triangles, std::vector<Pt> &points, Pt cut_O, Pt cut_N );

    // internal computation methods
    void                    mark_cut_faces_and_edges ( MarkCutInfo &mci, Node *node, TF sp );
    template<class F> void  for_each_triangle_rf     ( F &&func, TF max_ratio_area_error = 1e-1, bool remove_holes = true, std::mutex *m = 0 ) const; ///< for each triangle of the round faces
    void                    get_ap_edge_points       ( std::vector<Pt> &points, const Edge &edge, int nb_divs = 50, bool end = false ) const; ///<
    Pt                      point_for_angle          ( const Edge &edge, TF an ) const;
    TF                      angle                    ( const Edge &edge, Pt p ) const;
    TF                      area                     ( const Face &rp ) const;

    // helpers
    // void                 _make_ext_round_faces    ();
    void                    _get_centroid_planar     ( Pt &centroid, TF &area, const Face &fs ) const;
    void                    _get_centroid_round      ( Pt &centroid, TF &area, const Face &fs ) const;
};


} // namespace sdot

#include "ConvexPolyhedron3.tcc"



