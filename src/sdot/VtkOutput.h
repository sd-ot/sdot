#pragma once

#include "Point3.h"
#include "Point2.h"
#include <mutex>
#include <deque>

namespace sdot {

/**
  Class for simplified (and not optimized) vtk output
*/
template<int nb_cell_fields=0,class _TF=double>
class VtkOutput {
public:
    using                   TF               = _TF;
    using                   CN               = std::array<std::string,nb_cell_fields>;
    using                   CV               = std::array<TF,nb_cell_fields>;
    using                   PT               = Point3<TF>;
    using                   P2               = Point2<TF>;

    VtkOutput                                ( const CN &_cell_field_names = {} );

    void                    save             ( std::string filename ) const;
    void                    save             ( std::ostream &os ) const;

    void                    add_arc          ( PT center, PT A, PT B, PT tangent, const CV &cell_value = {}, unsigned n = 50 );
    void                    add_point        ( PT p, const CV &cell_value = {} );
    void                    add_point        ( P2 p, const CV &cell_value = {} );
    void                    add_lines        ( const std::vector<PT> &p, const CV &cell_value = {} );
    void                    add_lines        ( const std::vector<P2> &p, const CV &cell_value = {} );
    void                    add_arrow        ( PT center, PT dir, const CV &cell_value = {} );
    void                    add_circle       ( PT center, PT normal, TF radius, const CV &cell_value = {}, unsigned n = 50 );
    void                    add_polygon      ( const std::vector<PT> &p, const CV &cell_value = {} );

    std::mutex              mutex;

private:
    struct Pt {
        PT                  p;
        CV                  cell_values;
    };
    struct Li {
        std::vector<PT>     p;
        CV                  cell_values;
    };
    struct Po {
        std::vector<PT>     p;
        CV                  cell_values;
    };

    size_t                  _nb_vtk_cell_items() const;
    size_t                  _nb_vtk_points    () const;
    size_t                  _nb_vtk_cells     () const;

    CN                      _cell_field_names;
    std::vector<Po>         _polygons;
    std::deque<Pt>          _points;
    std::deque<Li>          _lines;
};

} // namespace sdot

#include "VtkOutput.tcc"
