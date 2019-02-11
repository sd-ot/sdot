#include "system/Assert.h"
#include "system/Math.h"
#include "VtkOutput.h"
#include <fstream>

template<int nb_cell_fields,class TF>
VtkOutput<nb_cell_fields,TF>::VtkOutput( const CN &_cell_field_names ) : _cell_field_names( _cell_field_names ) {
}

template<int nb_cell_fields,class TF>
void VtkOutput<nb_cell_fields,TF>::save( std::string filename ) const {
    std::ofstream os( filename.c_str() );
    save( os );
}

template<int nb_cell_fields,class TF>
void VtkOutput<nb_cell_fields,TF>::save( std::ostream &os ) const {
    os << "# vtk DataFile Version 3.0\n";
    os << "vtk output\n";
    os << "ASCII\n";
    os << "DATASET UNSTRUCTURED_GRID\n";

    // POINTS
    os << "POINTS " << _nb_vtk_points() << " float\n";
    for( Pt pt : _points )
        os << pt.p.x << " " << pt.p.y << " " << pt.p.z << "\n";
    for( Li li : _lines )
        for( PT pt : li.p )
            os << pt.x << " " << pt.y << " " << pt.z << "\n";
    for( Po li : _polygons )
        for( PT pt : li.p )
            os << pt.x << " " << pt.y << " " << pt.z << "\n";

    // CELL
    os << "CELLS " << _nb_vtk_cells() << " " << _nb_vtk_cell_items() << "\n";
    for( size_t i = 0; i < _points.size(); ++i )
        os << "1 " << i << "\n";
    size_t o = _points.size();
    for( Li li : _lines ) {
        os << li.p.size();
        for( size_t i = 0; i < li.p.size(); ++i )
            os << " " << o++;
        os << "\n";
    }
    for( Po li : _polygons ) {
        os << li.p.size();
        for( size_t i = 0; i < li.p.size(); ++i )
            os << " " << o++;
        os << "\n";
    }

    // CELL_TYPES
    os << "CELL_TYPES " << _nb_vtk_cells() << "\n";
    for( size_t i = 0; i < _points.size(); ++i )
        os << "1\n";
    for( size_t i = 0; i < _lines.size(); ++i )
        os << "4\n";
    for( size_t i = 0; i < _polygons.size(); ++i )
        os << "7\n";

    // CELL_DATA
    os << "CELL_DATA " << _nb_vtk_cells() << "\n";
    os << "FIELD FieldData " << _cell_field_names.size() << "\n";
    for( size_t num_field = 0; num_field < _cell_field_names.size(); ++num_field ) {
        os << _cell_field_names[ num_field ] << " 1 " << _nb_vtk_cells() << " float\n";
        for( const Pt &pt : _points )
            os << " " << pt.cell_values[ num_field ];
        for( const Li &li : _lines )
            os << " " << li.cell_values[ num_field ];
        for( const Po &li : _polygons )
            os << " " << li.cell_values[ num_field ];
        os << "\n";
    }
}

template<int nb_cell_fields,class TF>
void VtkOutput<nb_cell_fields,TF>::add_point( PT p, const CV &cell_value ) {
    _points.push_back( { p, cell_value } );
}

template<int nb_cell_fields,class TF>
void VtkOutput<nb_cell_fields,TF>::add_point( P2 p, const CV &cell_value ) {
    add_point( { p.x, p.y, 0.0 }, cell_value );
}

template<int nb_cell_fields,class TF>
void VtkOutput<nb_cell_fields,TF>::add_lines( const std::vector<PT> &p, const CV &cell_value ) {
    if ( p.size() < 2 )
        return;
    _lines.push_back( { p, cell_value } );
}

template<int nb_cell_fields,class TF>
void VtkOutput<nb_cell_fields,TF>::add_lines( const std::vector<P2> &p, const CV &cell_value ) {
    std::vector<PT> p3;
    for( const P2 &p2 : p )
        p3.push_back( { p2.x, p2.y, TF( 0 ) } );
    add_lines( p3, cell_value );
}

template<int nb_cell_fields,class TF>
void VtkOutput<nb_cell_fields,TF>::add_polygon( const std::vector<PT> &p, const CV &cell_values ) {
    if ( p.size() < 2 )
        return;
    _polygons.push_back( { p, cell_values } );
}

template<int nb_cell_fields,class TF>
void VtkOutput<nb_cell_fields,TF>::add_arc( PT C, PT A, PT B, PT tangent, const CV &cell_value, unsigned nb_divs ) {
    // add_lines( { A, A + tangent }, { 2 } );

    PT X = normalized( A - C ), Y = normalized( tangent );
    TF a = atan2p( dot( B - C, Y ), dot( B - C, X ) );

    std::vector<PT> pts;
    TF radius = norm_2( A - C );
    for( size_t i = 0, n = std::ceil( nb_divs * a / ( 2 * M_PI ) ); i <= n; ++i )
        pts.push_back( C + radius * cos( a * i / n ) * X + radius * sin( a * i / n ) * Y );
    add_lines( pts, cell_value );
}

template<int nb_cell_fields,class TF>
void VtkOutput<nb_cell_fields,TF>::add_circle( PT center, PT normal, TF radius, const CV &cell_value, unsigned n ) {
    PT X = normalized( ortho_rand( normal ) ), Y = normalized( cross_prod( normal, X ) );
    std::vector<PT> pts;
    for( size_t i = 0; i <= n; ++i )
        pts.push_back( center + radius * cos( 2 * M_PI * i / n ) * X + radius * sin( 2 * M_PI * i / n ) * Y );
    add_lines( pts, cell_value );
}

template<int nb_cell_fields,class TF>
void VtkOutput<nb_cell_fields,TF>::add_arrow( PT center, PT dir, const CV &cell_value ) {
    PT nd = ortho_rand( dir );
    add_lines( { center, center + dir, center + TF( 0.8 ) * dir + TF( 0.2 ) * nd }, cell_value );
    add_lines( { center + dir, center + TF( 0.8 ) * dir - TF( 0.2 ) * nd }, cell_value );
}

template<int nb_cell_fields,class TF>
size_t VtkOutput<nb_cell_fields,TF>::_nb_vtk_cell_items() const {
    size_t res = 2 * _points.size();
    for( Li li : _lines )
        res += 1 + li.p.size();
    for( Po po : _polygons )
        res += 1 + po.p.size();
    return res;
}

template<int nb_cell_fields,class TF>
size_t VtkOutput<nb_cell_fields,TF>::_nb_vtk_points() const {
    size_t res = _points.size();
    for( Li li : _lines )
        res += li.p.size();
    for( Po po : _polygons )
        res += po.p.size();
    return res;
}

template<int nb_cell_fields,class TF>
size_t VtkOutput<nb_cell_fields,TF>::_nb_vtk_cells() const {
    return _points.size() + _lines.size() + _polygons.size();
}
