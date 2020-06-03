// #include "Support/Math.h"
#include "VtkOutput.h"
#include "ASSERT.h"
#include <fstream>

namespace sdot {

VtkOutput::VtkOutput( const std::vector<std::string> &cell_fields_names ) {
    cell_fields.resize( cell_fields_names.size() );
    for( std::size_t i = 0; i < cell_fields_names.size(); ++i )
        cell_fields[ i ].name = cell_fields_names[ i ];
}


void VtkOutput::save( std::string filename ) const {
    std::ofstream os( filename.c_str() );
    save( os );
}


void VtkOutput::save( std::ostream &os ) const {
    os << "# vtk DataFile Version 3.0\n";
    os << "vtk output\n";
    os << "ASCII\n";
    os << "DATASET UNSTRUCTURED_GRID\n";

    // POINTS
    os << "POINTS " << _nb_vtk_points() << " float\n";
    for( SinglePoint pt : single_points )
        os << pt.p[ 0 ] << " " << pt.p[ 1 ] << " " << pt.p[ 2 ] << "\n";
    for( const Line &li : lines )
        for( Pt pt : li.p )
            os << pt[ 0 ] << " " << pt[ 1 ] << " " << pt[ 2 ] << "\n";
    for( const Polygon &li : polygons )
        for( Pt pt : li.p )
            os << pt[ 0 ] << " " << pt[ 1 ] << " " << pt[ 2 ] << "\n";

    // CELL
    os << "CELLS " << _nb_vtk_cells() << " " << _nb_vtk_cell_items() << "\n";
    for( size_t i = 0; i < single_points.size(); ++i )
        os << "1 " << i << "\n";
    size_t o = single_points.size();
    for( const Line &li : lines ) {
        os << li.p.size();
        for( size_t i = 0; i < li.p.size(); ++i )
            os << " " << o++;
        os << "\n";
    }
    for( const Polygon &li : polygons ) {
        os << li.p.size();
        for( size_t i = 0; i < li.p.size(); ++i )
            os << " " << o++;
        os << "\n";
    }

    // CELL_TYPES
    os << "CELL_TYPES " << _nb_vtk_cells() << "\n";
    for( size_t i = 0; i < single_points.size(); ++i )
        os << "1\n";
    for( size_t i = 0; i < lines.size(); ++i )
        os << "4\n";
    for( size_t i = 0; i < polygons.size(); ++i )
        os << "7\n";

    // CELL_DATA
    os << "CELL_DATA " << _nb_vtk_cells() << "\n";
    os << "FIELD FieldData " << cell_fields.size() << "\n";
    for( size_t num_field = 0; num_field < cell_fields.size(); ++num_field ) {
        os << cell_fields[ num_field ].name << " 1 " << _nb_vtk_cells() << " float\n";
        for( F v : cell_fields[ num_field ].v_points )
            os << " " << v;
        for( F v : cell_fields[ num_field ].v_lines )
            os << " " << v;
        for( F v : cell_fields[ num_field ].v_polygons )
            os << " " << v;
        os << "\n";
    }
}


void VtkOutput::add_point( Pt p, const std::vector<F> &cell_values ) {
    single_points.push_back( { p } );
    for( std::size_t i = 0; i < cell_fields.size(); ++i )
        cell_fields[ i ].v_points.push_back( i < cell_values.size() ? cell_values[ i ] : F( 0 ) );
}

void VtkOutput::append( const VtkOutput &vo ) {
    auto append = []( auto &a, const auto &b ) { a.insert( a.end(), b.begin(), b.end() ); };
    for( std::size_t i = 0; i < cell_fields.size(); ++i ) {
        append( cell_fields[ i ].v_lines , vo.cell_fields[ i ].v_lines    );
        append( cell_fields[ i ].v_points, vo.cell_fields[ i ].v_points   );
        append( cell_fields[ i ].v_points, vo.cell_fields[ i ].v_polygons );
    }
    append( lines   , vo.lines    );
    append( single_points  , vo.single_points   );
    append( polygons, vo.polygons );
}


//void VtkOutput::add_point( P2 p, const std::vector<TF> &cell_value ) {
//    add_point( { p.x, p.y, 0.0 }, cell_value );
//}


//void VtkOutput::add_lines( const std::vector<PT> &p, const CV &cell_value ) {
//    if ( p.size() < 2 )
//        return;
//    _lines.push_back( { p, cell_value } );
//}


//void VtkOutput::add_lines( const std::vector<P2> &p, const CV &cell_value ) {
//    std::vector<PT> p3;
//    for( const P2 &p2 : p )
//        p3.push_back( { p2.x, p2.y, TF( 0 ) } );
//    add_lines( p3, cell_value );
//}


void VtkOutput::add_polygon( const std::vector<Pt> &p, const std::vector<F> &cell_values ) {
    if ( p.size() < 2 )
        return;
    polygons.push_back( { p } );
    for( std::size_t i = 0; i < cell_fields.size(); ++i )
        cell_fields[ i ].v_polygons.push_back( i < cell_values.size() ? cell_values[ i ] : F( 0 ) );
}

void VtkOutput::add_lines( const std::vector<Pt> &p, const std::vector<F> &cell_values ) {
    lines.push_back( { p } );
    for( std::size_t i = 0; i < cell_fields.size(); ++i )
        cell_fields[ i ].v_lines.push_back( i < cell_values.size() ? cell_values[ i ] : F( 0 ) );
}


//void VtkOutput::add_arc( PT C, PT A, PT B, PT tangent, const CV &cell_value, unsigned nb_divs ) {
//    // add_lines( { A, A + tangent }, { 2 } );

//    PT X = normalized( A - C ), Y = normalized( tangent );
//    TF a = atan2p( dot( B - C, Y ), dot( B - C, X ) );

//    std::vector<PT> pts;
//    TF radius = norm_2( A - C );
//    for( size_t i = 0, n = std::ceil( nb_divs * a / ( 2 * M_PI ) ); i <= n; ++i )
//        pts.push_back( C + radius * cos( a * i / n ) * X + radius * sin( a * i / n ) * Y );
//    add_lines( pts, cell_value );
//}


//void VtkOutput::add_circle( PT center, PT normal, TF radius, const CV &cell_value, unsigned n ) {
//    PT X = normalized( ortho_rand( normal ) ), Y = normalized( cross_prod( normal, X ) );
//    std::vector<PT> pts;
//    for( size_t i = 0; i <= n; ++i )
//        pts.push_back( center + radius * cos( 2 * M_PI * i / n ) * X + radius * sin( 2 * M_PI * i / n ) * Y );
//    add_lines( pts, cell_value );
//}


//void VtkOutput::add_arrow( PT center, PT dir, const CV &cell_value ) {
//    PT nd = ortho_rand( dir );
//    add_lines( { center, center + dir, center + TF( 0.8 ) * dir + TF( 0.2 ) * nd }, cell_value );
//    add_lines( { center + dir, center + TF( 0.8 ) * dir - TF( 0.2 ) * nd }, cell_value );
//}


std::size_t VtkOutput::_nb_vtk_cell_items() const {
    size_t res = 2 * single_points.size();
    for( const Line &li : lines )
        res += 1 + li.p.size();
    for( const Polygon &po : polygons )
        res += 1 + po.p.size();
    return res;
}


std::size_t VtkOutput::_nb_vtk_points() const {
    size_t res = single_points.size();
    for( const Line &li : lines )
        res += li.p.size();
    for( const Polygon &po : polygons )
        res += po.p.size();
    return res;
}


std::size_t VtkOutput::_nb_vtk_cells() const {
    return single_points.size() + lines.size() + polygons.size();
}

} // namespace sdot
