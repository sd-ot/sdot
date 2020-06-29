#include "ConvexHull.h"

template<int dim,class TF>
ConvexHull<dim,TF>::ConvexHull( const std::vector<Pt> &pts, std::string friendly_name ) : friendly_name( friendly_name ), chi( pts ), pts( pts ) {
}

template<int dim,class TF>
bool ConvexHull<dim,TF>::is_a_permutation_of( const ConvexHull &that, TI *perm_this_to_that ) const {
    return chi.is_a_permutation_of( that.pts, that.chi, perm_this_to_that );
}

template<int dim,class TF>
void ConvexHull<dim,TF>::write_to_stream( std::ostream &os ) const {
    os << pts << "\n" << chi;
}


template<int dim,class TF>
void ConvexHull<dim,TF>::display_vtk( VtkOutput &vo, Pt offset ) const {
    std::vector<Pt> npts = pts;
    for( Pt &p : npts )
        p += offset;
    chi.display_vtk( vo, npts );
}

template<int dim,class TF>
TF ConvexHull<dim,TF>::measure() const {
    return chi.measure( pts );
}

template<int dim,class TF>
typename ConvexHull<dim,TF>::Pt ConvexHull<dim,TF>::center() const {
    return chi.center( pts );
}

template<int dim,class TF>
std::string ConvexHull<dim,TF>::name() const {
    if ( friendly_name.size() )
        return friendly_name;
    return "pouet" + std::to_string( pts.size() );
}

template<int dim,class TF>
ConvexHull<dim,TF> ConvexHull<dim,TF>::intersection( const ConvexHull &that ) const {
    ConvexHull ch = *this;
    that.for_each_normal( [&]( Pt center, Pt normal ) {
        ch = ch.cut( center, normal );
    } );
    return ch;
}

template<int dim,class TF>
ConvexHull<dim,TF> ConvexHull<dim,TF>::cut( Pt orig, Pt normal ) const {
    auto sp = [&]( TI n ) {
        return dot( pts[ n ] - orig, normal );
    };
    auto outside = [&]( TI n ) {
        return sp( n ) > 0;
    };

    std::vector<Pt> npts;
    for( TI i = 0; i < pts.size(); ++i )
        if ( ! outside( i ) )
            npts.push_back( pts[ i ] );

    std::vector<std::set<TI>> links;
    chi.get_links_rec( links );
    for( TI i = 0; i < links.size(); ++i )
        for( TI j : links[ i ] )
            if ( j >= i && outside( i ) != outside( j ) )
                npts.push_back( pts[ i ] + sp( i ) / ( sp( i ) - sp( j ) ) * ( pts[ j ] - pts[ i ] ) );

    return npts;
}

template<int dim,class TF>
void ConvexHull<dim,TF>::for_each_normal( const std::function<void(Pt,Pt)> &f ) const {
    for( const auto &next : chi.nexts )
        next.on_normal( pts, f );
}
