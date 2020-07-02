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
    using                        UserData            = typename Rp::UserData;
    using                        Vertex              = typename Rp::Vertex;
    enum {                       dim                 = Rp::dim };
    enum {                       nvi                 = nvi_ };
    using                        TF                  = typename Rp::TF;
    using                        TI                  = typename Rp::TI;

    using                        Face                = RecursivePolytopImpl<Rp,std::max(nvi-1,0)>;
    using                        Pt                  = Point<TF,dim>;
    using                        Pn                  = Point<TF,nvi>;

    /**/                         RecursivePolytopImpl();

    template<class F,int n> void for_each_item_rec   ( const F &fu, N<n> ) const;
    template<class F> void       for_each_item_rec   ( const F &fu, N<nvi> ) const;
    template<class F> void       for_each_item_rec   ( const F &fu ) const;
    static void                  add_convex_hull     ( IntrusiveList<RecursivePolytopImpl> &res, Rp &rp, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &normal, const Pt &center );
    void                         write_to_stream     ( std::ostream &os ) const;
    Vertex*                      first_vertex        () const;
    void                         plane_cut           ( IntrusiveList<RecursivePolytopImpl> &res, Rp &new_rp, const Rp &old_rp, std::vector<Vertex *> &new_vertices, IntrusiveList<Face> &io_faces, const Vertex *&io_vertex, const Pt &cut_normal, N<2> ) const;
    template<class B> void       plane_cut           ( IntrusiveList<RecursivePolytopImpl> &res, Rp &new_rp, const Rp &old_rp, std::vector<Vertex *> &new_vertices, IntrusiveList<Face> &io_faces, const Vertex *&io_vertex, const Pt &cut_normal, B ) const;
    TF                           measure             ( std::array<Pt,dim> &dirs, const Pt &prev_pt ) const;
    Pt                           center              () const; ///< a point in the "middle"

    Pt                           normal;             ///<
    IntrusiveList<Face>          faces;              ///<
    RecursivePolytopImpl*        next;               ///< used by parent->faces (IntrusiveList)
};

/** Edge */
template<class Rp>
struct RecursivePolytopImpl<Rp,1> {
    using                        UserData            = typename Rp::UserData;
    using                        Vertex              = typename Rp::Vertex;
    enum {                       dim                 = Rp::dim };
    enum {                       nvi                 = 1 };
    using                        TF                  = typename Rp::TF;
    using                        TI                  = typename Rp::TI;

    using                        Face                = Void;
    using                        Pt                  = Point<TF,dim>;
    using                        Pn                  = Point<TF,nvi>;

    /**/                         RecursivePolytopImpl();

    template<class F> void       for_each_item_rec   ( const F &fu, N<1> ) const;
    template<class F> void       for_each_item_rec   ( const F &fu ) const;
    static void                  add_convex_hull     ( IntrusiveList<RecursivePolytopImpl> &res, Rp &rp, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &normal, const Pt &center );
    void                         write_to_stream     ( std::ostream &os ) const;
    Vertex*                      first_vertex        () const;
    void                         plane_cut           ( IntrusiveList<RecursivePolytopImpl> &res, Rp &new_rp, const Rp &old_rp, std::vector<Vertex *> &new_vertices, IntrusiveList<Face> &io_faces, const Vertex *&io_vertex, const Pt &cut_normal, N<1> ) const;
    TF                           measure             ( std::array<Pt,dim> &dirs, const Pt &prev_pt ) const;
    Pt                           center              () const; ///< a point in the "middle"

    std::array<Vertex *,2>       vertices;           ///<
    Pt                           normal;             ///<
    RecursivePolytopImpl*        next;               ///< used by IntrusiveList
};

#include "RecursivePolytopImpl.tcc"

#endif // SDOT_SLOW_POLYTOP_HEADER
