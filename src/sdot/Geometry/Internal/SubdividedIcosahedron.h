#pragma once

#include "../Point3.h"
#include <map>

namespace sdot {

/**
*/
template<class TF>
class SubdividedIcosahedron {
public:
    using PT = Point3<TF>;

    struct Triangle {
        std::size_t P0, P1, P2; ///< connectivity
    };

    struct Mesh {
        TF                    max_area_error;
        std::vector<Triangle> triangles;
        std::vector<PT>       points;
    };

    SubdividedIcosahedron() {
        _make_first_mesh();
    }

    const Mesh &mesh_for_error( TF max_area_error ) {
        for( size_t num_item = 0; ; ++num_item ) {
            // need to add a new subdivision
            if ( num_item == _meshes.size() )
                _make_new_subdivision();
            //
            if ( _meshes[ num_item ].max_area_error <= max_area_error )
                return _meshes[ num_item ];
        }
    }

private:
    void _make_first_mesh() {
        Mesh item;
        item.max_area_error = std::numeric_limits<TF>::max();

        const TF t = 0.5 * ( 1 + std::sqrt( 5.0 ) );
        item.points.push_back( normalized( PT{ -1,  t,  0 } ) );
        item.points.push_back( normalized( PT{  1,  t,  0 } ) );
        item.points.push_back( normalized( PT{ -1, -t,  0 } ) );
        item.points.push_back( normalized( PT{  1, -t,  0 } ) );
        item.points.push_back( normalized( PT{  0, -1,  t } ) );
        item.points.push_back( normalized( PT{  0,  1,  t } ) );
        item.points.push_back( normalized( PT{  0, -1, -t } ) );
        item.points.push_back( normalized( PT{  0,  1, -t } ) );
        item.points.push_back( normalized( PT{  t,  0, -1 } ) );
        item.points.push_back( normalized( PT{  t,  0,  1 } ) );
        item.points.push_back( normalized( PT{ -t,  0, -1 } ) );
        item.points.push_back( normalized( PT{ -t,  0,  1 } ) );

        item.triangles.push_back( {  0, 11,  5 } );
        item.triangles.push_back( {  0,  5,  1 } );
        item.triangles.push_back( {  0,  1,  7 } );
        item.triangles.push_back( {  0,  7, 10 } );
        item.triangles.push_back( {  0, 10, 11 } );
        item.triangles.push_back( {  1,  5,  9 } );
        item.triangles.push_back( {  5, 11,  4 } );
        item.triangles.push_back( { 11, 10,  2 } );
        item.triangles.push_back( { 10,  7,  6 } );
        item.triangles.push_back( {  7,  1,  8 } );
        item.triangles.push_back( {  3,  9,  4 } );
        item.triangles.push_back( {  3,  4,  2 } );
        item.triangles.push_back( {  3,  2,  6 } );
        item.triangles.push_back( {  3,  6,  8 } );
        item.triangles.push_back( {  3,  8,  9 } );
        item.triangles.push_back( {  4,  9,  5 } );
        item.triangles.push_back( {  2,  4, 11 } );
        item.triangles.push_back( {  6,  2, 10 } );
        item.triangles.push_back( {  8,  6,  7 } );
        item.triangles.push_back( {  9,  8,  1 } );

        _meshes.push_back( std::move( item ) );
    }

    void _make_new_subdivision() {
        Mesh item;
        item.max_area_error = 0;

        std::map<std::pair<std::size_t,std::size_t>,std::size_t> edge_cuts;
        const Mesh &old_item = _meshes.back();
        item.points = old_item.points;
        for( const Triangle &tri : old_item.triangles ) {
            std::size_t mid_01 = _make_edge_cut( item.points, edge_cuts, tri.P0, tri.P1, normalized( TF( 0.5 ) * ( old_item.points[ tri.P0 ] + old_item.points[ tri.P1 ] ) ) );
            std::size_t mid_12 = _make_edge_cut( item.points, edge_cuts, tri.P1, tri.P2, normalized( TF( 0.5 ) * ( old_item.points[ tri.P1 ] + old_item.points[ tri.P2 ] ) ) );
            std::size_t mid_20 = _make_edge_cut( item.points, edge_cuts, tri.P2, tri.P0, normalized( TF( 0.5 ) * ( old_item.points[ tri.P2 ] + old_item.points[ tri.P0 ] ) ) );

            item.triangles.push_back( { tri.P0, mid_01, mid_20 } );
            item.triangles.push_back( { tri.P1, mid_12, mid_01 } );
            item.triangles.push_back( { tri.P2, mid_20, mid_12 } );
            item.triangles.push_back( { mid_01, mid_12, mid_20 } );

            item.max_area_error += std::abs(
                _tri_area( item.points[ tri.P0 ], item.points[ tri.P1 ], item.points[ tri.P2 ] ) -
                _tri_area( item.points[ tri.P0 ], item.points[ mid_01 ], item.points[ mid_20 ] ) -
                _tri_area( item.points[ tri.P1 ], item.points[ mid_12 ], item.points[ mid_01 ] ) -
                _tri_area( item.points[ tri.P2 ], item.points[ mid_20 ], item.points[ mid_12 ] ) -
                _tri_area( item.points[ mid_01 ], item.points[ mid_12 ], item.points[ mid_20 ] )
            );
        }

        item.max_area_error /= 4.0 / 3.0 * M_PI;
        _meshes.push_back( std::move( item ) );
    }

    static std::size_t _make_edge_cut( std::vector<PT> &pts, std::map<std::pair<std::size_t,std::size_t>,std::size_t> &edge_cuts, std::size_t P0, std::size_t P1, PT point ) {
        if ( P0 > P1 )
            std::swap( P0, P1 );
        auto edid = std::make_pair( P0, P1 );
        auto iter = edge_cuts.find( edid );
        if ( iter != edge_cuts.end() ) {
            std::size_t res = iter->second;
            edge_cuts.erase( iter );
            return res;
        }
        std::size_t res = pts.size();
        edge_cuts.emplace_hint( iter, std::make_pair( edid, res ) );
        pts.emplace_back( point );
        return res;
    }

    TF _tri_area( PT P0, PT P1, PT P2 ) {
        return norm_2_p2( cross_prod( P1 - P0, P2 - P0 ) ) / 2;
    }

    std::vector<Mesh> _meshes;
};

} // namespace sdot
