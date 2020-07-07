#ifndef SDOT_RECURSIVE_POLYTOP_HEADER
#define SDOT_RECURSIVE_POLYTOP_HEADER

#include "internal/RecursivePolytopVertex.h"
#include "internal/RecursivePolytopImpl.h"
#include <deque>

/**

*/
template<class TF_,int dim_,class TI_=std::size_t,class UserData_=Void>
class RecursivePolytop {
public:
    using                         UserData             = UserData_;
    using                         Vertex               = RecursivePolytopVertex<TF_,dim_,TI_,UserData>;
    enum {                        dim                  = dim_ };
    using                         TF                   = TF_;
    using                         TI                   = TI_;
    using                         Pt                   = Point<TF,dim>;
    using                         Rp                   = RecursivePolytop;

    /**/                          RecursivePolytop     ( std::initializer_list<Pt> pts );
    /**/                          RecursivePolytop     ( const std::vector<Pt> &pts );
    /**/                          RecursivePolytop     ( TI nb_vertices = 0 );

    TI                            nb_vertices          () const { return vertices.size(); }
    TI                            nb_faces             () const;
    const Vertex&                 vertex               ( TI i ) const { return vertices[ i ]; }
    Vertex&                       vertex               ( TI i ) { return vertices[ i ]; }

    bool                          all_vertices_are_used() const;
    template<class F,int n> void  for_each_item_rec    ( const F &fu, N<n> ) const; ///< for a fixed nvi
    template<class F> void        for_each_item_rec    ( const F &fu ) const;
    bool                          valid_vertex_prop    ( const std::vector<Pt> &pts ) const; ///< check for planarity and rank of faces
    bool                          can_use_perm_pts     ( const Pt *pts, TI *num_in_pts, bool want_convexity = true ) const; ///< true if can find a permutation of pts such as taking it as argument to with_points would lead to a valid shape
    void                          make_convex_hull     ();
    void                          write_to_stream      ( std::ostream &os, std::string nl = "\n  ", std::string ns = "  " ) const;
    void                          update_normals       (); ///< TODO: check orientation
    template<class VO> void       display_vtk          ( VO &vo ) const;
    Rp                            with_points          ( const std::vector<Pt> &pts ) const;
    bool                          is_convex            () const;
    RecursivePolytop              plane_cut            ( Pt orig, Pt normal, const std::function<UserData(const UserData &,const UserData &,TF,TF)> &nf = {} ) const;
    bool                          contains             ( const Pt &pt ) const;
    TF                            measure              () const;

    static TF                     measure_intersection ( const Rp &a, const Rp &b ); ///< intersect this with the faces of that
    static void                   get_intersections    ( std::deque<std::array<Rp,2>> &res, const Rp &a, const Rp &b ); ///< intersect this with the faces of that

private:
    template<class R,int n>       friend               struct RecursivePolytopImpl;
    using                         Impl                 = RecursivePolytopImpl<RecursivePolytop,dim>;

    template<class Rpi> void      make_tmp_connections ( const IntrusiveList<Rpi> &items ) const; ///< vertex.beg and vertex.end will be used to get offets in this->tmp_connections
    template<class Rpi> TI        get_connected_graphs ( const IntrusiveList<Rpi> &items ) const;
    void                          mark_connected_rec   ( const Vertex *v ) const;
    TI                            num_graph            ( const Vertex *v ) const;

    BumpPointerPool               pool;                ///< (to be defined before vertices)
    mutable TI                    date;                ///< for graph operations
    IntrusiveList<Impl>           impls;               ///< faces
    FsVec<Vertex>                 vertices;            ///<
    mutable std::vector<bool>     tmp_edges;           ///<
    mutable std::vector<Vertex *> tmp_connections;     ///<
};

#include "RecursivePolytop.tcc"

#endif // SDOT_RECURSIVE_POLYTOP_HEADER
