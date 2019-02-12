#pragma once

#include "SubdividedIcosahedron.h"
#include "FunctionEnum.h"
#include "VtkOutput.h"
#include <functional>
#include <algorithm>

/**
  TF = floating point type
  CI = information to store at each cut

  Beware: if there's a need for a sphere_cut, musts be done AFTER the plane_cuts.
*/
template<class Pc,class CI>
class ConvexPolyhedron3 {
public:
    static constexpr bool     allow_ball_cut = Pc::allow_ball_cut;
    using                     TI               = typename Pc::TI; ///< index type
    using                     TF               = typename Pc::TF; ///< index type
    using                     Pt               = Point3<TF>;      ///< 3D point
    struct                    Box              { Pt p0, p1; };

    struct Node {
        union                 SoI              { TF sd; TI index; };
        void                  write_to_stream  ( std::ostream &os ) const { os << pos; }
        inline bool           inside           () { return soi.sd < 0; }

        Pt                    pos; ///< position
        SoI                   soi; ///< signed dist to the plane of the last cut
    };

    struct Edge {
        using                 NL               = std::array<TI,2>;

        void                  write_to_stream  ( std::ostream &os ) const { if ( straight() ) os << "L(" << n0 << "," << n1 << ")"; else os << "R(" << n0 << "," << n1 << ",ci=" << cut_index << ")"; }
        Edge&                 add_node_off     ( TI off, TI cio ) { n0 += off; n1 += off; cut_index += cio; return *this; }
        bool                  straight         () const { return radius < 0; }
        bool                  round            () const { return radius >= 0; }
        Pt                    Y                () const { return tangent_0; }

        // input data
        TI                    cut_index;  ///<
        TI                    n0;         ///< 1st node
        TI                    n1;         ///< 2nd node

        // computed data
        Pt                    tangent_0;  ///< tangent in n0
        Pt                    tangent_1;  ///< tangent in n1
        TF                    angle_1;    ///< angle of n1 (angle of n0 = 0)
        Pt                    center;     ///<
        TF                    radius;     ///<
        Pt                    X;          ///< X axis (normalized( n0 - center ))

        // tmp computed data
        TI                    nedge;      ///< results of the previous cut
        bool                  used;
    };

    struct FlatSurface {
        void                  write_to_stream  ( std::ostream &os ) const { os << "FS(be=" << beg_in_edge_indices << ",ee=" << end_in_edge_indices << ",ci=" << cut_index << ")"; }
        TI                    size             () const { return end_in_edge_indices - beg_in_edge_indices; }

        TI                    beg_in_edge_indices;
        TI                    end_in_edge_indices;
        TI                    cut_index;
    };

    struct RoundSurface {
        void                  write_to_stream  ( std::ostream &os ) const { os << "RS(be=" << beg_in_edge_indices << ",ee=" << end_in_edge_indices << ")"; }

        TI                    beg_in_edge_indices;
        TI                    end_in_edge_indices;
    };

    struct CutInfo {
        CI                    cut_id;   ///< provided by the user (as argument of the cut function)
        Pt                    cut_O;    ///< a point in the cut plane
        Pt                    cut_N;    ///< normal, oriented toward exterior (removed part)
        TI                    used;
    };

    struct Hole {
        void                  write_to_stream  ( std::ostream &os ) const { os << cut_index; }

        TI                    cut_index;   ///<
    };

    /// we start from a tetrahedra that includes the sphere defined by sphere_center and sphere_radius... but this sphere is not used
    /**/                      ConvexPolyhedron3        ( Pt englobing_center = { 0, 0, 0 }, TF englobing_radius = 1, CI englobing_cut_id = {} );
    /**/                      ConvexPolyhedron3        ( const Box &box, CI cut_id = {} );

    // information
    void                      for_each_boundary_measure( FunctionEnum::Unit, const std::function<void( TF area, CI id )> &f ) const;
    void                      write_to_stream          ( std::ostream &os ) const;
    void                      for_each_node            ( const std::function<void( Pt v )> &f ) const;
    TI                        nb_points                () const { return nodes.size(); }
    Pt                        point                    ( TI n ) const { return nodes[ n ].pos; }
    template<class V> void    display                  ( V &vo, const typename V::CV &cell_data = {}, bool filled = true, TF max_ratio_area_error = 1e-1, bool display_tangents = false ) const;

    // modifications
    void                      set_cut_ids              ( CI cut_id ); ///< replace all the cut_ids
    template<int no> void     plane_cut                ( Pt origin, Pt normal, CI cut_id, N<no> normal_is_normalized ); ///< return true if effective cut
    void                      plane_cut                ( Pt origin, Pt normal, CI cut_id = {} ) { plane_cut( origin, normal, cut_id, N<1>() ); }
    void                      ball_cut                 ( Pt center, TF radius, CI cut_id = {} ); ///< beware: only one sphere cut is authorized, and it must be done after all the plane cuts.
    void                      clear                    ( Pt englobing_center, TF englobing_radius, CI englobing_cut_id = {} );
    void                      clear                    ( const Box &box, CI cut_id = {} );

    // computations
    void                      add_centroid_contrib     ( FunctionEnum::Unit, Pt &ctd, TF &vol ) const;
    TF                        boundary_measure         ( FunctionEnum::Unit ) const;
    Pt                        centroid                 ( FunctionEnum::Unit ) const;
    TF                        measure                  ( FunctionEnum::Unit ) const;

    void                      add_centroid_contrib     ( Pt &ctd, TF &vol ) const { return add_centroid_contrib( FunctionEnum::Unit(), ctd, vol ); }
    TF                        boundary_measure         ()                   const { return boundary_measure    ( FunctionEnum::Unit()           ); }
    Pt                        centroid                 ()                   const { return centroid            ( FunctionEnum::Unit()           ); }
    TF                        measure                  ()                   const { return measure             ( FunctionEnum::Unit()           ); }

    // tests
    bool                      contains                 ( const Pt &pos ) const;
    bool                      empty                    () const { return nodes.empty(); }

    // approximate computations
    template<class Fu> TF     boundary_measure_ap      ( const Fu &fu, TF max_ratio_area_error = 1e-4 ) const; ///< area from a triangulation of the surface
    template<class Fu> Pt     centroid_ap              ( const Fu &fu, TI n = 1e8 ) const;                     ///< centroid, computed with monte-carlo
    template<class Fu> TF     measure_ap               ( const Fu &fu, TI n = 1e8 ) const;                     ///< volume, computed with monte-carlo

    TF                        boundary_measure_ap      ( TF max_ratio_area_error = 1e-4 ) const { return boundary_measure_ap( FunctionEnum::Unit(), max_ratio_area_error ); } ///< area from a triangulation of the surface
    Pt                        centroid_ap              ( TI n = 1e8 )                     const { return centroid_ap        ( FunctionEnum::Unit(), n                    ); } ///< centroid, computed with monte-carlo
    TF                        measure_ap               ( TI n = 1e8 )                     const { return measure_ap         ( FunctionEnum::Unit(), n                    ); } ///< volume, computed with monte-carlo

    CI                        sphere_cut_id;
    Pt                        sphere_center;
    TF                        sphere_radius;

private:
    // internal modifications methods
    void                      add_round_surface        ( const std::vector<TI> &edges );
    void                      add_flat_surface         ( std::pair<TI,TI> edge_indices_bounds, TI cut_index );

    TI                        add_straight_edge        ( TI n0, TI n1, TI cut_index );
    std::pair<TI,TI>          add_edge_indices         ( TI e0, TI e1, TI e2, TI e3 );
    std::pair<TI,TI>          add_edge_indices         ( TI e0, TI e1, TI e2 );
    TI                        add_round_edge           ( TI n0, TI n1, TI cut_index );
    TI                        add_cut_info             ( Pt cut_O, Pt cut_N, CI cut_id );
    TI                        add_node                 ( Pt pos );

    template<class Triangle>
    static void               p_cut                    ( std::vector<Triangle> &triangles, std::vector<Pt> &points, Pt cut_O, Pt cut_N );

    // internal computation methods
    template<class F> void    for_each_triangle_rf     ( F &&func, TF max_ratio_area_error = 1e-1, bool remove_holes = true, std::mutex *m = 0 ) const; ///< for each triangle of the round faces
    void                      remove_unused_nodes      ( TI old_nodes_size );
    void                      remove_unused_edges      ( TI old_edges_size, bool new_edges_are_round );
    void                      remove_unused_cuts       ();
    void                      get_edge_points          ( std::vector<Pt> &points, const Edge &edge, int nb_divs = 50, bool end = false ) const;
    Pt                        point_for_angle          ( const Edge &edge, TF an ) const { return edge.center + edge.radius * std::cos( an ) * edge.X + edge.radius * std::sin( an ) * edge.Y(); }
    TF                        angle                    ( const Edge &edge, Pt p ) const;
    TF                        area                     ( const RoundSurface &rp ) const;
    TF                        area                     ( const FlatSurface &fp ) const;

    // helpers
    void                      _make_ext_round_faces    ();
    static void               _get_connected_points    ( std::vector<bool> &connected, const std::vector<std::vector<TI>> &connected_points, TI index );
    void                      _get_connections_rec     ( TI nb_connections, TI num_node );
    void                      _get_centroid_rf         ( Pt &centroid, TF &area ) const;
    static TI                 _make_edge_cut           ( std::vector<Pt> &pts, std::map<std::pair<TI,TI>,TI> &edge_cuts, TI P0, TI P1, Pt point );
    void                      _get_centroid            ( Pt &centroid, TF &area, const FlatSurface &fs ) const;

    // attributes
    std::vector<RoundSurface> part_round_surfaces;
    std::vector<FlatSurface>  flat_surfaces;
    std::vector<TI>           edge_indices;
    std::vector<CutInfo>      cut_info;
    std::vector<Edge>         edges;
    std::vector<Node>         nodes;
    std::vector<Hole>         holes;

    std::vector<TI>           old_edges_indices;
    std::vector<TI>           node_connectivity;
    std::vector<TI>           num_connections;
    TI                        nb_connections;
};

#include "ConvexPolyhedron3.tcc"



