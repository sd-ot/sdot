#ifndef SDOT_SLOW_POLYTOP_HEADER
#define SDOT_SLOW_POLYTOP_HEADER

#include "../../support/BumpPointerPool.h"
#include "../../support/IntrusiveList.h"
#include "../../support/FsVec.h"
#include "../Point.h"

/**
*/
template<class Rp,int nvi_>
struct RecursivePolytopImpl {
    enum {                    rp_nvi              = Rp::nvi };
    enum {                    dim                 = Rp::dim };
    enum {                    nvi                 = nvi_ };
    using                     TF                  = typename Rp::TF;
    using                     TI                  = typename Rp::TI;

    using                     Vertex              = typename Rp::Vertex;
    using                     Face                = RecursivePolytopImpl<Rp,std::max(nvi-1,0)>;
    using                     Pt                  = Point<TF,dim>;
    using                     Pn                  = Point<TF,nvi>;

    /**/                      RecursivePolytopImpl();

    template<class Fu> void   for_each_item_rec   ( const Fu &fu ) const;
    void                      add_convex_hull     ( BumpPointerPool &pool, FsVec<Vertex> vertices, TI *indices, TI nb_indices, N<0> );
    template<class B> void    add_convex_hull     ( BumpPointerPool &pool, FsVec<Vertex> vertices, TI *indices, TI nb_indices, B );
    void                      write_to_stream     ( std::ostream &os ) const;
    static void               make_base_dirs      ( std::array<Pt,nvi> &base_dirs, FsVec<Vertex> vertices, const TI *indices, TI nb_indices );
    Pn                        proj                ( const Pt &pt ) const;

    // std::array<Pt,rp_nvi-nvi> prev_centers;
    std::array<Pt,nvi>        base_dirs;
    FsVec<Vertex *>           vertices;
    Pt                        normal;
    IntrusiveList<Face>       faces;
    Pt                        orig;
    RecursivePolytopImpl*     next;

};

#include "RecursivePolytopImpl.tcc"

#endif // SDOT_SLOW_POLYTOP_HEADER
