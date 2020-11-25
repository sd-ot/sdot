#include "VtkOutput.h"
#include <fstream>

namespace sdot {

VtkOutput::VtkOutput( std::vector<std::string> cell_field_names, std::vector<std::string> node_field_names ) {
    for( std::string name : cell_field_names )
        cell_fields.push_back( { name, {} } );
    for( std::string name : node_field_names )
        node_fields.push_back( { name, {} } );
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
    os << "POINTS " << points.size() << " float\n";
    for( const Pt &pt : points )
        os << pt[ 0 ] << " " << pt[ 1 ] << " " << pt[ 2 ] << "\n";

    // CELL
    os << "CELLS " << cell_types.size() << " " << cell_items.size() << "\n";
    for( TI v : cell_items )
        os << v << "\n";

    // CELL_TYPES
    os << "CELL_TYPES " << cell_types.size() << "\n";
    for( TI v : cell_types )
        os << v << "\n";

    // CELL_DATA
    os << "CELL_DATA " << cell_types.size() << "\n";
    os << "FIELD FieldData " << cell_fields.size() << "\n";
    for( size_t num_field = 0; num_field < cell_fields.size(); ++num_field ) {
        os << cell_fields[ num_field ].name << " 1 " << cell_types.size() << " float\n";
        for( TF v : cell_fields[ num_field ].values )
            os << " " << v;
        os << "\n";
    }
}

void VtkOutput::add_triangle( const std::array<Pt,3> &pts ) {
    add_item( pts.data(), pts.size(), 5 );
}

void VtkOutput::add_pyramid( const std::array<Pt,5> &pts ) {
    std::array<Pt,5> npts{ pts[ 0 ], pts[ 1 ], pts[ 3 ], pts[ 2 ], pts[ 4 ] };
    add_item( npts.data(), npts.size(), 14 );
}

void VtkOutput::add_polygon( const std::vector<Pt> &pts ) {
    add_item( pts.data(), pts.size(), 7 );
}

void VtkOutput::add_wedge( const std::array<Pt,6> &pts ) {
    add_item( pts.data(), pts.size(), 13 );
}

void VtkOutput::add_tetra( const std::array<Pt,4> &pts ) {
    add_item( pts.data(), pts.size(), 10 );
}

void VtkOutput::add_quad( const std::array<Pt,4> &pts ) {
    add_item( pts.data(), pts.size(), 9 );
}

void VtkOutput::add_hexa( const std::array<Pt,8> &pts ) {
    add_item( pts.data(), pts.size(), 12 );
}

void VtkOutput::add_lines( const std::vector<Pt> &pts ) {
    add_item( pts.data(), pts.size(), 4 );
}

void VtkOutput::add_item( const Pt *pts_data, TI pts_size, TI vtk_type, const std::vector<TF> &cell_data, const std::vector<TF> &node_data ) {
    TI os = points.size();
    for( TI i = 0; i < pts_size; ++i )
        points.push_back( pts_data[ i ] );

    cell_items.push_back( pts_size );
    for( TI i = 0; i < pts_size; ++i )
        cell_items.push_back( os++ );

    cell_types.push_back( vtk_type );

    for( TI i = 0, m = std::min( cell_data.size(), cell_fields.size() ); i < m; ++i )
        cell_fields[ i ].values.push_back( cell_data[ i ] );
    for( TI i = cell_data.size(); i < cell_fields.size(); ++i )
        cell_fields[ i ].values.push_back( 0 );

    for( TI i = 0, m = std::min( node_data.size(), node_fields.size() ); i < m; ++i )
        node_fields[ i ].values.push_back( node_data[ i ] );
    for( TI i = node_data.size(); i < node_fields.size(); ++i )
        node_fields[ i ].values.push_back( 0 );
}

}
