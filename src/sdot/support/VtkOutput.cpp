#include "VtkOutput.h"
#include <fstream>

VtkOutput::VtkOutput() {
}

void VtkOutput::save( std::string filename ) const {
    std::ofstream os( filename.c_str() );
    save( os );
}

void VtkOutput::save(std::ostream &os) const {
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

    //    // CELL_DATA
    //    os << "CELL_DATA " << _nb_vtk_cells() << "\n";
    //    os << "FIELD FieldData " << cell_fields.size() << "\n";
    //    for( size_t num_field = 0; num_field < cell_fields.size(); ++num_field ) {
    //        os << cell_fields[ num_field ].name << " 1 " << _nb_vtk_cells() << " float\n";
    //        for( TF v : cell_fields[ num_field ].v_points )
    //            os << " " << v;
    //        for( TF v : cell_fields[ num_field ].v_lines )
    //            os << " " << v;
    //        for( TF v : cell_fields[ num_field ].v_polygons )
    //            os << " " << v;
    //        for( TF v : cell_fields[ num_field ].v_tetras )
    //            os << " " << v;
    //        os << "\n";
    //    }
}

void VtkOutput::add_triangle( std::array<Pt,3> pts ) {
    add_item( pts.data(), pts.size(), 5 );
}

void VtkOutput::add_pyramid( std::array<Pt,5> pts ) {
    std::array<Pt,5> npts{ pts[ 0 ], pts[ 1 ], pts[ 3 ], pts[ 2 ], pts[ 4 ] };
    add_item( npts.data(), npts.size(), 14 );
}

void VtkOutput::add_polygon( const std::vector<Pt> &pts ) {
    add_item( pts.data(), pts.size(), 7 );
}

void VtkOutput::add_wedge( std::array<Pt,6> pts ) {
    add_item( pts.data(), pts.size(), 13 );
}

void VtkOutput::add_tetra( std::array<Pt,4> pts ) {
    add_item( pts.data(), pts.size(), 10 );
}

void VtkOutput::add_quad( std::array<Pt,4> pts ) {
    add_item( pts.data(), pts.size(), 9 );
}

void VtkOutput::add_hexa( std::array<Pt,8> pts ) {
    add_item( pts.data(), pts.size(), 12 );
}

void VtkOutput::add_line( const std::vector<Pt> &pts ) {
    add_item( pts.data(), pts.size(), 4 );
}

void VtkOutput::add_item( const Pt *pts_data, TI pts_size, TI vtk_type ) {
    TI os = points.size();
    for( TI i = 0; i < pts_size; ++i )
        points.push_back( pts_data[ i ] );

    cell_items.push_back( pts_size );
    for( TI i = 0; i < pts_size; ++i )
        cell_items.push_back( os++ );

    cell_types.push_back( vtk_type );
}

