#include "GenCuts.h"
#include <fstream>

template<int dim>
void make_cuts( GlobalGenCutData &gcd, int max_nb_point_circle = 4 ) {
    GenCuts<dim> gc( gcd );

    // make the base shapes
    if ( dim == 1 ) {
        gc.add_ref_shape( "" ); // a simple segment
    } else {
        for( int n = 3; n <= std::min( 4, max_nb_point_circle ); ++n ) {
            for( int k = 0; k < ( 1 << ( dim - 2 ) ); ++k ) {
                std::ostringstream name;
                name << n;
                for( int d = 0; d < dim - 2; ++d )
                    name << ( k & ( 1 << d ) ? "E" : "S" );
                gc.add_ref_shape( name.str() );
            }
        }
        for( int n = 5; n <= max_nb_point_circle; ++n ) {
            std::ostringstream name;
            name << n;
            for( int d = 0; d < dim - 2; ++d )
                name << "S";
            gc.add_ref_shape( name.str() );
        }
    }

    // -> cut_cases
    std::ofstream fcu( "src/sdot/geometry/internal/generated/SetOfElementaryPolytops_cut_cases_" + std::to_string( dim ) + "D.h" );
    for( std::size_t num_ref_shape = 0; num_ref_shape < gc.ref_shapes.size(); ++num_ref_shape ) {
        gc.setup_cut_nodes_for( gc.ref_shapes[ num_ref_shape ] );
        gc.setup_parts_from_cut_nodes();
        P( gc.parts.size() );

        gc.make_best_combs_from_parts();
        gc.makes_comb_for_cases();

        gc.write_code_for_cases( fcu );
    }

    // -> VecOps
    std::ofstream fop( "src/sdot/geometry/internal/generated/SetOfElementaryPolytopsVecOps_" + std::to_string( dim ) + "D.h" );
    gc.write_cut_op_funcs( fop );

    // -> nb_vertices
    std::ofstream fmv( "src/sdot/geometry/internal/generated/SetOfElementaryPolytops_max_nb_vertices_" + std::to_string( dim ) + "D.h" );
    fmv << "if ( dim == " << dim << " ) return " << gc.max_nb_vertices_per_elem() << ";\n";

    // -> nb_vertices
    std::ofstream fnv( "src/sdot/geometry/internal/generated/SetOfElementaryPolytops_nb_vertices_" + std::to_string( dim ) + "D.h" );
    for( const auto &rs : gc.ref_shapes )
        fnv << "if ( dim == " << dim << " && name == \"" << rs.name << "\" ) return " << rs.rp.nb_vertices() << ";\n";

    // -> nb_faces
    std::ofstream fnf( "src/sdot/geometry/internal/generated/SetOfElementaryPolytops_nb_faces_" + std::to_string( dim ) + "D.h" );
    for( const auto &rs : gc.ref_shapes )
        fnf << "if ( dim == " << dim << " && name == \"" << rs.name << "\" ) return " << rs.rp.nb_faces() << ";\n";

    // -> measure
    std::ofstream fme( "src/sdot/geometry/internal/generated/SetOfElementaryPolytops_measure_" + std::to_string( dim ) + "D.h" );
    gc.write_measure_func( fme );
}

/*
  Prop: pour toutes les coupes,
    * on cherche si ça correspond à un volume élémentaire
*/
int main( int, char **argv ) {
    GlobalGenCutData gcd;
    if ( std::string( argv[ 1 ] ) == "2" )
        make_cuts<2>( gcd, 6 );
    if ( std::string( argv[ 1 ] ) == "3" )
        make_cuts<3>( gcd, 4 );
}

