#pragma once

#include "Point.h"
#include <vector>
#include <array>

namespace sdot {

/***/
class VtkOutput {
public:
    enum {                   VtkPoint    = 1   };
    enum {                   VtkLine     = 4   };
    enum {                   VtkPoly     = 7   };
    enum {                   VtkTetra    = 10  };

    using                    TF          = double;
    using                    TI          = std::size_t;
    using                    Pt          = Point<TF,3>;

    struct                   FieldData   { std::string name; std::vector<TF> values; };

    /**/                     VtkOutput   ( std::vector<std::string> cell_field_names = {}, std::vector<std::string> node_field_names = {} );

    void                     save        ( std::string filename ) const;
    void                     save        ( std::ostream &os ) const;

    void                     add_polygon ( const std::vector<Pt>  &pts );
    void                     add_lines   ( const std::vector<Pt>  &pts );

    void                     add_triangle( const std::array<Pt,3> &pts );
    void                     add_pyramid ( const std::array<Pt,5> &pts );
    void                     add_wedge   ( const std::array<Pt,6> &pts );
    void                     add_tetra   ( const std::array<Pt,4> &pts );
    void                     add_hexa    ( const std::array<Pt,8> &pts );
    void                     add_quad    ( const std::array<Pt,4> &pts );

    void                     add_item    ( const Pt *pts_data, TI pts_size, TI vtk_type, const std::vector<TF> &cell_data = {}, const std::vector<TF> &node_data = {} );

    std::vector<Pt>          points;
    std::vector<TI>          cell_types;
    std::vector<TI>          cell_items;
    std::vector<FieldData>   cell_fields;
    std::vector<FieldData>   node_fields;
};

}
