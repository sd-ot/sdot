#include "../src/sdot/geometry/SetOfElementaryPolytops.h"
#include "../src/sdot/geometry/Point.h"

#include "../src/sdot/kernels/KernelSlot.h"

#include "../src/sdot/support/P.h"

#include <cmath>

using namespace sdot;

using Pt = Point<double,2>;
using TI = std::uint64_t;
using TF = Pt::TF;

void test_triangle() {
    std::vector<std::unique_ptr<KernelSlot>> ak = KernelSlot::available_slots( TypeName<double>::name(), TypeName<std::size_t>::name() );
    KernelSlot *ks = ak[ 0 ].get();

    SetOfElementaryPolytops sp( ks, 2 );

    // construct
    TI nb_triangles = 15;
    sp.add_repeated( triangle(), nb_triangles,
        { ks, std::vector<TF>{ 0, 0, 1, 0, 0, 1 } }, // positions
        { ks, std::vector<TI>{ 0, 1, 2 } } // face ids
    );
    P( sp );

    // cut
    std::vector<TF> cxs, cys, css;
    std::vector<TI> new_face_ids;
    for( std::size_t i = 0; i < nb_triangles; ++i ) {
        TF a = 2 * M_PI * i / nb_triangles;
        Pt p = { std::cos( a ), std::sin( a ) };
        cxs.push_back( p[ 0 ] );
        cys.push_back( p[ 1 ] );
        css.push_back( dot( p, Pt{ 0.33, 0.33 } ) );
        new_face_ids.push_back( 100 + i );
    }

    sp.plane_cut( { { ks, cxs }, { ks, cys } }, { ks, css }, { ks, new_face_ids } );
    P( sp );

    // display
    std::vector<VtkOutput::Pt> off_vtk;
    for( std::size_t i = 0; i < nb_triangles; ++i )
        off_vtk.push_back( VtkOutput::Pt{ 0.0, 0.0, 0.2 * i } );

    VtkOutput vo;
    sp.display_vtk( vo, off_vtk.data() );
    vo.save( "cut.vtk" );
}

void test_quad() {
    std::vector<std::unique_ptr<KernelSlot>> ak = KernelSlot::available_slots( TypeName<double>::name(), TypeName<std::size_t>::name() );
    KernelSlot *ks = ak[ 0 ].get();

    SetOfElementaryPolytops sp( ks, 2 );

    // construct
    sp.add_repeated( quad(), 1,
        { ks, std::vector<TF>{ 0, -1, 1, 1, 0, -2, -1, 1 } }, // positions
        { ks, std::vector<TI>{ 0, 1, 2, 3 } }, // face ids
        0 // beg item id
    );
    //    sp.add_repeated( quad(), 1,
    //        { ks, std::vector<TF>{ 0, -1, 1, 1, 0, -2, -1, 1 } }, // positions
    //        { ks, std::vector<TI>{ 0, 1, 2, 3 } } // face ids
    //    );
    P( sp );


    // cut
    std::vector<TF> cxs{ 0 }, cys{ 1 }, css{ 0 };
    std::vector<TI> new_face_ids{ 10 };

    sp.plane_cut( { { ks, cxs }, { ks, cys } }, { ks, css }, { ks, new_face_ids } );
    P( sp );

    // display
    VtkOutput vo;
    sp.display_vtk( vo );
    vo.save( "cut.vtk" );
}

int main() {
    test_triangle();
    //    test_quad();
}
