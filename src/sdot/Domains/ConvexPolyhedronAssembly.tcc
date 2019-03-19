#include "../Integration/SpaceFunctions/Constant.h"
#include "ConvexPolyhedronAssembly.h"

namespace sdot {

template<class Pc>
void ConvexPolyhedronAssembly<Pc>::add_convex_polyhedron( const std::vector<Pt> &positions, const std::vector<Pt> &normals, TF coeff, TI cut_id ) {
    englobing_polyheron_is_up_to_date = false;

    using std::min;
    using std::max;
    Pt min_pos;
    Pt max_pos;
    Pt delta;
    for( std::size_t d = 0; d < dim; ++d ) {
        min_pos[ d ] = + std::numeric_limits<TF>::max();
        max_pos[ d ] = - std::numeric_limits<TF>::max();
        for( std::size_t i = 0; i < positions.size(); ++i ) {
            min_pos[ d ] = min( min_pos[ d ], positions[ i ][ d ] );
            max_pos[ d ] = max( max_pos[ d ], positions[ i ][ d ] );
        }
        delta[ d ] = max_pos[ d ] - min_pos[ d ];
    }

    Item item;
    item.coeff = coeff;

    item.polyhedron = { typename CP::Box{ min_pos - delta, max_pos + delta }, cut_id };
    for( std::size_t i = 0; i < positions.size(); ++i )
        item.polyhedron.plane_cut( positions[ i ], normalized( normals[ i ] ), cut_id );

    items.push_back( std::move( item ) );
}

template<class Pc>
void ConvexPolyhedronAssembly<Pc>::add_box( Pt p0, Pt p1, TF coeff, TI cut_id ) {
    englobing_polyheron_is_up_to_date = false;

    items.push_back( { { typename CP::Box{ p0, p1 }, cut_id }, coeff } );
}

template<class Pc>
void ConvexPolyhedronAssembly<Pc>::normalize() {
    if ( TF mea = measure() )
        for( Item &item : items )
            item.coeff /= mea;
}

template<class Pc>
const typename ConvexPolyhedronAssembly<Pc>::CP& ConvexPolyhedronAssembly<Pc>::englobing_convex_polyhedron() const {
    using std::min;
    using std::max;

    if ( items.size() == 1 )
        return items[ 0 ].polyhedron;

    if ( englobing_polyheron_is_up_to_date == false ) {
        mutex.lock();
        if ( englobing_polyheron_is_up_to_date == false ) {
            englobing_polyheron_is_up_to_date = true;

            Pt delta;
            Pt min_pos;
            Pt max_pos;
            for( std::size_t d = 0; d < dim; ++d ) {
                min_pos[ d ] = + std::numeric_limits<TF>::max();
                max_pos[ d ] = - std::numeric_limits<TF>::max();
            }
            for( const Item &item : items ) {
                min_pos = min( min_pos, item.polyhedron.min_position() );
                max_pos = max( max_pos, item.polyhedron.max_position() );
            }
            for( std::size_t d = 0; d < dim; ++d )
                delta[ d ] = max_pos[ d ] - min_pos[ d ];

            englobing_polyheron = typename CP::Box{ min_pos - delta, max_pos + delta };
        }
        mutex.unlock();
    }

    return englobing_polyheron;
}

template<class Pc>
typename ConvexPolyhedronAssembly<Pc>::Pt ConvexPolyhedronAssembly<Pc>::min_position() const {
    Pt res;
    for( std::size_t d = 0; d < dim; ++d )
        res[ d ] = + std::numeric_limits<TF>::max();
    for( const Item &item : items )
        res = min( res, item.polyhedron.min_position() );
    return res;
}

template<class Pc>
typename ConvexPolyhedronAssembly<Pc>::Pt ConvexPolyhedronAssembly<Pc>::max_position() const {
    Pt res;
    for( std::size_t d = 0; d < dim; ++d )
        res[ d ] = - std::numeric_limits<TF>::max();
    for( const Item &item : items )
        res = max( res, item.polyhedron.max_position() );
    return res;
}

template<class Pc>
typename ConvexPolyhedronAssembly<Pc>::TF ConvexPolyhedronAssembly<Pc>::measure() const {
    TF res = 0;
    for( const Item &item : items )
        res += item.coeff * item.polyhedron.measure();
    return res;
}

template<class Pc>
typename ConvexPolyhedronAssembly<Pc>::TF ConvexPolyhedronAssembly<Pc>::coeff_at( const Pt &pos ) const {
    TF res = 0;
    for( const Item &item : items )
        if ( item.polyhedron.contains( pos ) )
            res += item.coeff;
    return res;
}

template<class Pc> template<class F>
void ConvexPolyhedronAssembly<Pc>::for_each_intersection( CP &cp, const F &f ) const {
    if ( items.size() == 1 )
        return f( cp, SpaceFunctions::Constant<TF>{ items[ 0 ].coeff } );

    CP ccp;
    for( const Item &item : items ) {
        ccp = item.polyhedron;
        ccp.intersect_with( cp );
        f( ccp, SpaceFunctions::Constant<TF>{ item.coeff } );
    }
}

template<class Pc> template<class V>
void ConvexPolyhedronAssembly<Pc>::display_boundaries( V &vtk_output ) const {
    for( const Item &item : items )
        item.polyhedron.display( vtk_output, { item.coeff }, false );
}

template<class Pc> template<class V>
void ConvexPolyhedronAssembly<Pc>::display_coeffs( V &vtk_output ) const {
    for( const Item &item : items )
        item.polyhedron.display( vtk_output, { item.coeff }, true );
}

} // namespace sdot
