#ifndef SDOT_RECURSIVE_POLYTOP_CONNECTIVITY_HEADER
#define SDOT_RECURSIVE_POLYTOP_CONNECTIVITY_HEADER

#include "../../support/Void.h"
#include "../Point.h"
#include <vector>

/**
*/
template<class TF_,int dim_,int nvi_=dim_,class TI_=std::size_t>
struct RecursivePolytopConnectivity {
    enum {                         dim              = dim_ };
    enum {                         nvi              = nvi_ };
    using                          TF               = TF_;
    using                          TI               = TI_;

    using                          Face             = RecursivePolytopConnectivity<TF,dim,std::max(nvi-1,0),TI>;
    using                          Pt               = Point<TF,dim>;
    using                          Pn               = Point<TF,nvi>;

    static void                    add_convex_hull  ( std::vector<RecursivePolytopConnectivity> &res, const Pt *points, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &normal, const Pt &center );

    template<class F,int n> void   for_each_item_rec( const F &fu, N<n> ) const;
    template<class F> void         for_each_item_rec( const F &fu, N<nvi> ) const;
    template<class F> void         for_each_item_rec( const F &fu ) const;
    void                           write_to_stream  ( std::ostream &os, bool rec = true ) const;
    TI                             first_vertex     () const;
    void                           conn_cut         ( std::vector<RecursivePolytopConnectivity> &res, TI &nb_points, TI *new_points_per_edge, std::vector<bool> &outside );
    bool                           contains         ( const Pt *points, const Pt &pt ) const;
    TF                             measure          ( const Pt *points, std::array<Pt,dim> &dirs, const Pt &prev_pt ) const;
    Pt                             center           ( const Pt *points ) const; ///< a point in the "middle"

    Pt                             normal;          ///< (optionnal)
    std::vector<Face>              faces;           ///<
};

/** Edge */
template<class TF_,int dim_,class TI_>
struct RecursivePolytopConnectivity<TF_,dim_,1,TI_> {
    enum {                         dim              = dim_ };
    enum {                         nvi              = 1 };
    using                          TF               = TF_;
    using                          TI               = TI_;

    using                          Face             = Void;
    using                          Pt               = Point<TF,dim>;
    using                          Pn               = Point<TF,nvi>;

    static void                    add_convex_hull  ( std::vector<RecursivePolytopConnectivity> &res, const Pt *points, TI *indices, TI nb_indices, Pt *normals, Pt *dirs, const Pt &normal, const Pt &center );

    template<class F> void         for_each_item_rec( const F &fu, N<1> ) const;
    template<class F> void         for_each_item_rec( const F &fu ) const;
    void                           write_to_stream  ( std::ostream &os, bool rec = true ) const;
    TI                             first_vertex     () const;
    void                           conn_cut         ( std::vector<RecursivePolytopConnectivity> &res, TI &nb_points, TI *new_points_per_edge, std::vector<bool> &outside );
    bool                           contains         ( const Pt *points, const Pt &pt ) const;
    TF                             measure          ( const Pt *points, std::array<Pt,dim> &dirs, const Pt &prev_pt ) const;
    Pt                             center           ( const Pt *points ) const; ///< a point in the "middle"

    std::array<TI,2>               vertices;        ///<
    Pt                             normal;          ///<
};

#include "RecursivePolytopConnectivity.tcc"

#endif // SDOT_RECURSIVE_POLYTOP_CONNECTIVITY_HEADER
