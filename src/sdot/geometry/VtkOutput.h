#pragma once

#include "Point.h"
#include <vector>
#include <array>

namespace sdot {

/***/
class VtkOutput {
public:
    enum {          VtkPoint    = 1   };
    enum {          VtkLine     = 4   };
    enum {          VtkPoly     = 7   };
    enum {          VtkTetra    = 10  };

    using           TF          = double;
    using           TI          = std::size_t;
    using           Pt          = Point<TF,3>;

    /**/            VtkOutput   ();

    void            save        ( std::string filename ) const;
    void            save        ( std::ostream &os ) const;

    void            add_polygon ( const std::vector<Pt>  &pts );
    void            add_lines   ( const std::vector<Pt>  &pts );

    void            add_triangle( const std::array<Pt,3> &pts );
    void            add_pyramid ( const std::array<Pt,5> &pts );
    void            add_wedge   ( const std::array<Pt,6> &pts );
    void            add_tetra   ( const std::array<Pt,4> &pts );
    void            add_hexa    ( const std::array<Pt,8> &pts );
    void            add_quad    ( const std::array<Pt,4> &pts );

    void            add_item    ( const Pt *pts_data, TI pts_size, TI vtk_type );

    std::vector<Pt> points;
    std::vector<TI> cell_types;
    std::vector<TI> cell_items;
};

inline void VtkOutput::add_item( const Pt *pts_data, TI pts_size, TI vtk_type ) {
    TI os = points.size();
    for( TI i = 0; i < pts_size; ++i )
        points.push_back( pts_data[ i ] );

    cell_items.push_back( pts_size );
    for( TI i = 0; i < pts_size; ++i )
        cell_items.push_back( os++ );

    cell_types.push_back( vtk_type );
}

}
