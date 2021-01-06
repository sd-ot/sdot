#ifndef SDOT_POINT_H
#define SDOT_POINT_H

// #include <parex/TensorOrder.h>
// #include <parex/Math.h>
#include <ostream>
#include <vector>

namespace sdot {

/**
*/
template<class TF_,int dim_>
struct Point {
    enum {                          dim            = dim_ };
    using                           TF             = TF_;

    template<class TG>              Point          ( const Point<TG,dim> &p );
    template<class TG>              Point          ( const TG *v );
    template<class TG>              Point          ( TG x, TG y, TG z, TG w );
    template<class TG>              Point          ( TG x, TG y, TG z );
    template<class TG>              Point          ( TG x, TG y );
    /**/                            Point          ( TF x );
    /**/                            Point          ();

    // IO
    void                            write_to_stream( std::ostream &os ) const { for( int i = 0; i < dim; ++i ) os << ( i ? "," : "" ) << data[ i ]; }
    template<class Bq> static Point read_from      ( Bq &bq );
    template<class Bq> void         write_to       ( Bq &bq ) const;

    // accessors
    const TF&                       operator[]     ( std::size_t d ) const { return data[ d ]; }
    TF&                             operator[]     ( std::size_t d ) { return data[ d ]; }
    const TF*                       begin          () const { return data; }
    const TF*                       end            () const { return data + dim; }

    bool                            operator<      ( const Point &that ) const;
    operator                        bool           () const;

    TF                              data[ dim ];
};

// template<class TF,int dim> struct TensorOrder<Point<TF,dim>> { enum { value = TensorOrder<TF>::value + 1 }; };

template<class TF,int dim> TF                         norm_2_p2 ( Point<TF,dim> p );
template<class TF,int dim> TF                         norm_2    ( Point<TF,dim> p );
template<class TF,int dim> TF                         dot       ( Point<TF,dim> a, Point<TF,dim> b );
template<class TF,int dim> Point<TF,dim>&             operator+=( Point<TF,dim> &a, Point<TF,dim> b );
template<class TF,int dim> Point<TF,dim>&             operator-=( Point<TF,dim> &a, Point<TF,dim> b );
template<class TF,int dim> Point<TF,dim>&             operator/=( Point<TF,dim> &a, TF b );
template<class TF,int dim> Point<TF,dim>&             operator*=( Point<TF,dim> &a, TF b );
template<class TF,int dim> Point<TF,dim>              operator+ ( Point<TF,dim> a, Point<TF,dim> b );
template<class TF,int dim> Point<TF,dim>              operator+ ( Point<TF,dim> a, TF b );
template<class TF,int dim> Point<TF,dim>              operator- ( Point<TF,dim> a, Point<TF,dim> b );
template<class TF,int dim> Point<TF,dim>              operator- ( Point<TF,dim> a, TF b );
template<class TF,int dim> Point<TF,dim>              operator- ( Point<TF,dim> a );
template<class TF,int dim> Point<TF,dim>              operator* ( TF m, Point<TF,dim> p );
template<class TF,int dim> Point<TF,dim>              operator* ( Point<TF,dim> a, Point<TF,dim> b );
template<class TF,int dim> Point<TF,dim>              operator/ ( Point<TF,dim> p, TF d );
template<class TF,int dim> Point<TF,dim>              operator/ ( Point<TF,dim> p, Point<TF,dim> d );
template<class TF,int dim> bool                       operator==( Point<TF,dim> p, Point<TF,dim> q );
template<class TF,int dim> bool                       operator!=( Point<TF,dim> p, Point<TF,dim> q );
template<class TF,int dim> Point<TF,dim>              min       ( Point<TF,dim> p, Point<TF,dim> q );
template<class TF,int dim> Point<TF,dim>              max       ( Point<TF,dim> p, Point<TF,dim> q );
template<class TF,int dim> TF                         max       ( Point<TF,dim> p );
template<class TF,int dim> TF                         has_nan   ( Point<TF,dim> p );
template<class TF,int dim> Point<TF,dim>              normalized( Point<TF,dim> p, TF a = 1e-40 );
template<class TF>         Point<TF,2>                cross_prod( Point<TF,2> p );
template<class TF>         Point<TF,3>                cross_prod( Point<TF,3> a, Point<TF,3> b );
template<class TF,int dim> Point<TF,dim>              cross_prod( const Point<TF,dim> *pts );
template<class TF>         Point<TF,3>                ortho_rand( Point<TF,3> a );
template<class TF,int dim> Point<TF,dim>              ortho_norm( Point<TF,dim> D, Point<TF,dim> N );
template<class TF,int dim> std::vector<Point<TF,dim>> base_from ( const std::vector<Point<TF,dim>> &dirs );
template<class TF,int dim> int                        rank_pts  ( const std::vector<Point<TF,dim>> &dirs );

}

#include "Point.tcc"

#endif // SDOT_POINT_H
