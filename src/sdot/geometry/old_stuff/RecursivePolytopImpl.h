#ifndef SDOT_SLOW_POLYTOP_HEADER
#define SDOT_SLOW_POLYTOP_HEADER

#include "../../support/BumpPointerPool.h"
#include "../../support/IntrusiveList.h"
#include "../../support/FsVec.h"
#include "../../support/Void.h"
#include "../Point.h"

/**
*/
template<class Rp,int nvi_>
struct RecursivePolytopImpl {
    using                          UserData             = typename Rp::UserData;
    using                          Vertex               = typename Rp::Vertex;
    enum {                         dim                  = Rp::dim };
    enum {                         nvi                  = nvi_ };
    using                          TF                   = typename Rp::TF;
    using                          TI                   = typename Rp::TI;

    using                          Face                 = RecursivePolytopImpl<Rp,std::max(nvi-1,0)>;
    using                          Pt                   = Point<TF,dim>;
    using                          Pn                   = Point<TF,nvi>;

    /**/                           RecursivePolytopImpl ();

    TI                             make_unique_vertices ( const Vertex *vertices ) const; ///< store in vertices[ i ]->tmp_v for v in in 0..size an unique list of vertices
    template<class F,int n> void   for_each_item_rec    ( const F &fu, N<n> ) const;
    template<class F> void         for_each_item_rec    ( const F &fu, N<nvi> ) const;
    template<class F> void         for_each_item_rec    ( const F &fu ) const;
    bool                           valid_vertex_prop    ( const std::vector<Pt> &pts ) const;
    static void                    add_convex_hull      ( IntrusiveList<RecursivePolytopImpl> &res, Rp &rp, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &normal, const Pt &center );
    void                           write_to_stream      ( std::ostream &os ) const;
    template<class F> void         for_each_vertex      ( const F &fu ) const;
    void                           update_normals       ( Pt *normals, const Vertex *vertices, TI *indices, const Pt &center );
    Vertex*                        first_vertex         () const;
    template<class R,class V> void with_points          ( IntrusiveList<R> &res, BumpPointerPool &pool, V *new_vertices ) const;
    void                           plane_cut            ( IntrusiveList<RecursivePolytopImpl> &res, Rp &new_rp, const Rp &old_rp, std::vector<Vertex *> &new_vertices, IntrusiveList<Face> &io_faces, Vertex *&oi_vertices, const Pt &cut_normal, N<2> ) const;
    template<class B> void         plane_cut            ( IntrusiveList<RecursivePolytopImpl> &res, Rp &new_rp, const Rp &old_rp, std::vector<Vertex *> &new_vertices, IntrusiveList<Face> &io_faces, Vertex *&oi_vertices, const Pt &cut_normal, B ) const;
    bool                           contains             ( const Pt &pt ) const;
    TF                             measure              ( std::array<Pt,dim> &dirs, const Pt &prev_pt ) const;
    Pt                             center               () const; ///< a point in the "middle"

    Pt                             normal;              ///<
    IntrusiveList<Face>            faces;               ///<
    RecursivePolytopImpl*          next;                ///< used by parent->faces (IntrusiveList)
};

/** Edge */
template<class Rp>
struct RecursivePolytopImpl<Rp,1> {
    using                          UserData             = typename Rp::UserData;
    using                          Vertex               = typename Rp::Vertex;
    enum {                         dim                  = Rp::dim };
    enum {                         nvi                  = 1 };
    using                          TF                   = typename Rp::TF;
    using                          TI                   = typename Rp::TI;

    using                          Face                 = Void;
    using                          Pt                   = Point<TF,dim>;
    using                          Pn                   = Point<TF,nvi>;

    /**/                           RecursivePolytopImpl ();

    void                           for_each_intersection( const Pt &pos, const Pt &dir, const std::function<void( TF alpha, Pt inter )> &f ) const;
    template<class F> void         for_each_item_rec    ( const F &fu, N<1> ) const;
    template<class F> void         for_each_item_rec    ( const F &fu ) const;
    bool                           valid_vertex_prop    ( const std::vector<Pt> &pts ) const;
    static void                    add_convex_hull      ( IntrusiveList<RecursivePolytopImpl> &res, Rp &rp, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &normal, const Pt &center );
    void                           write_to_stream      ( std::ostream &os ) const;
    template<class F> void         for_each_vertex      ( const F &fu ) const;
    void                           update_normals       ( Pt *normals, const Vertex *vertices, TI *indices, const Pt &center );
    Vertex*                        first_vertex         () const;
    template<class R,class V> void with_points          ( IntrusiveList<R> &res, BumpPointerPool &pool, V *new_vertices ) const;
    void                           plane_cut            ( IntrusiveList<RecursivePolytopImpl> &res, Rp &new_rp, const Rp &old_rp, std::vector<Vertex *> &new_vertices, IntrusiveList<Face> &io_faces, Vertex *&oi_vertices, const Pt &cut_normal, N<1> ) const;
    bool                           contains             ( const Pt &pt ) const;
    TF                             measure              ( std::array<Pt,dim> &dirs, const Pt &prev_pt ) const;
    Pt                             center               () const; ///< a point in the "middle"

    std::array<Vertex *,2>         vertices;            ///<
    Pt                             normal;              ///<
    RecursivePolytopImpl*          next;                ///< used by IntrusiveList
};

#include "RecursivePolytopImpl.tcc"

#endif // SDOT_SLOW_POLYTOP_HEADER
