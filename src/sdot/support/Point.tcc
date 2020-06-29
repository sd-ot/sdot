#include "Point.h"
#include "Conv.h"

template<class TF,int dim> template<class TG>
Point<TF,dim>::Point( const Point<TG,dim> &p ) {
    for( int i = 0; i < dim; ++i )
        data[ i ] = conv( p.data[ i ], S<TF>() );
}

template<class TF,int dim> template<class TG>
Point<TF,dim>::Point( const TG *v ) {
    for( int i = 0; i < dim; ++i )
        data[ i ] = TF( v[ i ] );
}

template<class TF,int dim> template<class TG>
Point<TF,dim>::Point( TG x, TG y, TG z ) {
    data[ 0 ] = TF( x );
    data[ 1 ] = TF( y );
    data[ 2 ] = TF( z );
}

template<class TF,int dim> template<class TG>
Point<TF,dim>::Point( TG x, TG y ) {
    data[ 0 ] = TF( x );
    data[ 1 ] = TF( y );
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

template<class TF,int dim>
bool Point<TF,dim>::operator<( const Point &that ) const {
    for( int i = 0; i < dim; ++i )
        if ( data[ i ] != that.data[ i ] )
            return data[ i ] < that.data[ i ];
    return false;
}

