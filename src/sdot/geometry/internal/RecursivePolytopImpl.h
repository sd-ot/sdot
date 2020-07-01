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
    enum {                    dim                    = Rp::dim };
    enum {                    nvi                    = nvi_ };
    using                     TF                     = typename Rp::TF;
    using                     TI                     = typename Rp::TI;

    using                     Vertex                 = typename Rp::Vertex;
    using                     Face                   = RecursivePolytopImpl<Rp,std::max(nvi-1,0)>;
    using                     Pt                     = Point<TF,dim>;
    using                     Pn                     = Point<TF,nvi>;

    /**/                      RecursivePolytopImpl   ();

    void                      make_vertices_from_face( BumpPointerPool &pool, TI &date );
    template<class Fu> void   for_each_item_rec      ( const Fu &fu ) const;
    void                      add_convex_hull        ( BumpPointerPool &pool, FsVec<Vertex> vertices, TI *indices, TI nb_indices, Pt *prev_normals, TI &date, N<0> );
    template<class B> void    add_convex_hull        ( BumpPointerPool &pool, FsVec<Vertex> vertices, TI *indices, TI nb_indices, Pt *prev_normals, TI &date, B );
    void                      write_to_stream        ( std::ostream &os ) const;
    void                      sort_vertices          ( std::array<Pt,dim> &dirs, N<1> );
    template<class B> void    sort_vertices          ( std::array<Pt,dim> &dirs, B );
    void                      plane_cut              ( BumpPointerPool &pool, IntrusiveList<RecursivePolytopImpl> &new_rps, std::vector<Vertex *> &new_vertices, TI date, N<1> ) const;
    template<class B> void    plane_cut              ( BumpPointerPool &pool, IntrusiveList<RecursivePolytopImpl> &new_rps, std::vector<Vertex *> &new_vertices, TI date, B ) const;
    TF                        measure                ( std::array<Pt,dim> &dirs, N<1> ) const;
    template<class B> TF      measure                ( std::array<Pt,dim> &dirs, B ) const;
    Pn                        proj                   ( const Pt &pt ) const;

    // std::array<Pt,rp_nvi-nvi> prev_centers;
    FsVec<Vertex *>           vertices;
    Pt                        normal;
    Pt                        center;
    IntrusiveList<Face>       faces;
    RecursivePolytopImpl*     next;

};

#include "RecursivePolytopImpl.tcc"

#endif // SDOT_SLOW_POLYTOP_HEADER
