#pragma once

#include "Point.h"
#include <vector>
#include <deque>

namespace sdot {

/**
  Class for simplified (and not optimized) vtk output
*/
class VtkOutput {
public:
    using                    F                 = double;
    using                    Pt                = Point<F,3>;

    /**/                     VtkOutput         ( const std::vector<std::string> &cell_fields_names = {} );

    //    void               add_arc           ( Pt center, PT A, PT B, PT tangent, const CV &cell_value = {}, unsigned n = 50 );
    //    void               add_lines         ( const std::vector<P2> &p, const CV &cell_value = {} );
    //    void               add_arrow         ( PT center, PT dir, const CV &cell_value = {} );
    //    void               add_circle        ( PT center, PT normal, TF radius, const CV &cell_value = {}, unsigned n = 50 );
    void                     add_polygon       ( const std::vector<Pt> &p, const std::vector<F> &cell_values = {} );
    void                     add_lines         ( const std::vector<Pt> &p, const std::vector<F> &cell_values = {} );
    void                     add_point         ( Pt p, const std::vector<F> &cell_values = {} );

    void                     append            ( const VtkOutput &vo );

    void                     save              ( std::string filename ) const;
    void                     save              ( std::ostream &os ) const;

private:
    struct                   Field             { std::string name; std::vector<F> v_polygons, v_points, v_lines; };

    struct                   Polygon           { std::vector<Pt> p; };
    struct                   SinglePoint       { Pt              p; };
    struct                   Line              { std::vector<Pt> p; };

    std::size_t              _nb_vtk_cell_items() const;
    std::size_t              _nb_vtk_points    () const;
    std::size_t              _nb_vtk_cells     () const;

    std::deque<SinglePoint>  single_points;
    std::vector<Field>       cell_fields;
    std::deque<Polygon>      polygons;
    std::deque<Line>         lines;
};

} // namespace sdot
