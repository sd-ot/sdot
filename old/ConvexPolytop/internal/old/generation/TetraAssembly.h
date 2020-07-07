#include <vector>
#include <array>

#include "../../support/VtkOutput.h"
#include "../../support/Rational.h"

/**/
struct TetraAssembly {
    using              TF              = Rational;
    using              TI              = std::size_t;
    using              Pt              = Point<TF,3>;
    struct             Tetra           { std::array<Pt,4> pts; };
    enum {             dim             = 3 };

    void               add_intersection( const TetraAssembly &a, const TetraAssembly &b );
    void               add_pyramid     ( std::array<Pt,5> pts );
    void               add_tetra       ( std::array<Pt,4> pts );
    void               add_wedge       ( std::array<Pt,6> pts );
    void               add_hexa        ( std::array<Pt,8> pts );

    void               plane_cut       ( Pt pos, Pt dir );

    static TF          measure_tetra   ( const Pt *pts );
    void               display_vtk     ( VtkOutput &vo ) const;
    TF                 measure         () const;

    std::vector<Tetra> tetras;
};
