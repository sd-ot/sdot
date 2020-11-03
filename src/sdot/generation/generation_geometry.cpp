#include "NamedRecursivePolytop.h"
#include "GlobGeneGeomData.h"
#include "../support/P.h"
#include <fstream>

//// nsmake cpp_flag -std=c++17

using namespace sdot;

using TI = std::size_t;
using TF = Rational;

void init_of_primitive_shapes( std::vector<NamedRecursivePolytop> &primitive_shapes ) {
    // 2D
    for( TI nb_pts_circle = 3; nb_pts_circle < 7; ++nb_pts_circle ) {
        std::vector<std::vector<TF>> points;
        for( TI i = 0; i < nb_pts_circle; ++i ) {
            double a = 2 * M_PI * i / nb_pts_circle;
            points.push_back( {
                int( std::round( 2520 * cos( a ) ) ) / TF( 1000 ),
                int( std::round( 2520 * sin( a ) ) ) / TF( 1000 )
            } );
        }
        primitive_shapes.push_back( { RecursivePolytop::convex_hull( points ), "S" + std::to_string( nb_pts_circle ) } );
    }
}

int main() {
    std::vector<NamedRecursivePolytop> primitive_shapes;
    init_of_primitive_shapes( primitive_shapes );


    // summary.h
    std::string shape_directory = "src/sdot/geometry/shape_types/";
    std::ofstream summary( shape_directory + "summary.h" );
    summary << "// generated file\n";
    summary << "#pragma once\n";
    summary << "#include \"../ShapeType.h\"\n";
    summary << "namespace sdot {\n\n";

    // shape.cpp + summary.h
    GlobGeneGeomData gggd;
    for( const NamedRecursivePolytop &ps : primitive_shapes ) {
        std::ofstream incl( shape_directory + ps.name + ".h" );
        std::ofstream impl( shape_directory + ps.name + ".cpp" );
        ps.write_primitive_shape_incl( incl );
        ps.write_primitive_shape_impl( impl, gggd, primitive_shapes );

        // summary.h
        summary << "#include \"" << ps.name << ".h\"\n";
    }
    summary << "\n}\n";

    // KernelSlot_gen_decl.h
    gggd.write_gen_defs( "src/sdot/kernels/variants/KernelSlot_gen_def_cpu.h", false );
    gggd.write_gen_defs( "src/sdot/kernels/variants/KernelSlot_gen_def_gpu.h", true );
    gggd.write_gen_decls( "src/sdot/kernels/KernelSlot_gen_decl.h" );
}
