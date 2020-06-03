#include "Point.h"

namespace sdot {

template<class TF,int dim> template<class TG>
Point<TF,dim>::Point( const Point<TG,dim> &p ) {
    for( int i = 0; i < dim; ++i )
        data[ i ] = p.data[ i ];
}

template<class TF,int dim>
Point<TF,dim>::Point( const TF *v ) {
    for( int i = 0; i < dim; ++i )
        data[ i ] = v[ i ];
}

template<class TF,int dim>
Point<TF,dim>::Point( TF x, TF y, TF z ) {
    data[ 0 ] = x;
    data[ 1 ] = y;
    data[ 2 ] = z;
}

template<class TF,int dim>
Point<TF,dim>::Point( TF x, TF y ) {
    data[ 0 ] = x;
    data[ 1 ] = y;
}

template<class TF,int dim>
Point<TF,dim>::Point( TF x ) {
    for( int i = 0; i < dim; ++i )
        data[ i ] = x;
}

template<class TF,int dim>
Point<TF,dim>::Point() {
}

// IO
template<class TF,int dim> template<class Bq>
Point<TF,dim> Point<TF,dim>::read_from( Bq &bq )  {
    Point<TF,dim> res;
    for( int i = 0; i < dim; ++i )
        res.data[ i ] = bq.read();
    return res;
}

template<class TF,int dim> template<class Bq>
void Point<TF,dim>::write_to( Bq &bq ) const {
    for( int i = 0; i < dim; ++i )
        bq << data[ i ];
}

} // namespace sdot
