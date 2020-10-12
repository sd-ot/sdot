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

    void            add_triangle( std::array<Pt,3> pts );
    void            add_polygon ( const std::vector<Pt> &pts );
    void            add_pyramid ( std::array<Pt,5> pts );
    void            add_wedge   ( std::array<Pt,6> pts );
    void            add_tetra   ( std::array<Pt,4> pts );
    void            add_hexa    ( std::array<Pt,8> pts );
    void            add_quad    ( std::array<Pt,4> pts );
    void            add_line    ( const std::vector<Pt> &pts );

    void            add_item    ( const Pt *pts_data, TI pts_size, TI vtk_type );

    std::vector<Pt> points;
    std::vector<TI> cell_types;
    std::vector<TI> cell_items;
};

}
