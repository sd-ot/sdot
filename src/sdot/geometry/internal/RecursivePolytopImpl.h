#ifndef SDOT_SLOW_POLYTOP_HEADER
#define SDOT_SLOW_POLYTOP_HEADER

#include "../../support/BumpPointerPool.h"
#include "../../support/IntrusiveList.h"
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
    static void                  add_convex_hull     ( IntrusiveList<RecursivePolytopImpl> &res, BumpPointerPool &pool, Vertex *vertices, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &normal, const Pt &center );
    void                         write_to_stream     ( std::ostream &os ) const;
    void                         update_center       ();
    void                         plane_cut           ( IntrusiveList<RecursivePolytopImpl> &res, BumpPointerPool &pool, std::vector<Vertex *> &new_vertices, IntrusiveList<Face> &io_faces, const Vertex *&io_vertex, N<2> ) const;
    template<class B> void       plane_cut           ( IntrusiveList<RecursivePolytopImpl> &res, BumpPointerPool &pool, std::vector<Vertex *> &new_vertices, IntrusiveList<Face> &io_faces, const Vertex *&io_vertex, B ) const;
    TF                           measure             ( std::array<Pt,dim> &dirs ) const;

    Pt                           center;             ///<
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
    static void                  add_convex_hull     ( IntrusiveList<RecursivePolytopImpl> &res, BumpPointerPool &pool, Vertex *vertices, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &normal, const Pt &center );
    void                         write_to_stream     ( std::ostream &os ) const;
    void                         plane_cut           ( IntrusiveList<RecursivePolytopImpl> &res, BumpPointerPool &pool, std::vector<Vertex *> &new_vertices, IntrusiveList<Face> &io_faces, const Vertex *&io_vertex, N<1> ) const;
    TF                           measure             ( std::array<Pt,dim> &dirs ) const;

    std::array<Vertex *,2>       vertices;           ///<
    Pt                           center;             ///<
    Pt                           normal;             ///<
    RecursivePolytopImpl*        next;               ///< used by IntrusiveList
};

#include "RecursivePolytopImpl.tcc"

#endif // SDOT_SLOW_POLYTOP_HEADER
