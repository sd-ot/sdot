#include "GenCuts.h"

template<int dim>
void make_cuts( GlobalGenCutData &gcd, int max_nb_point_circle = 4 ) {
    GenCuts<dim> gc( gcd );

    if ( dim == 1 ) {
        gc.add_ref_shape( "" );
    } else {
        for( int n = 3; n <= max_nb_point_circle; ++n ) {
            for( int k = 0; k < ( 1 << dim - 2 ); ++k ) {
                std::ostringstream name;
                name << n;
                for( int d = 0; d < dim - 2; ++d )
                    name << ( k & ( 1 << d ) ? "E" : "S" );
                gc.add_ref_shape( name.str() );
            }
        }
    }

    gc.ref_shapes.resize( 1 );

    gc.setup_cut_nodes_for( gc.ref_shapes[ 0 ] );
    gc.setup_parts_from_cut_nodes();
    // gc.display_parts();
    P( gc.parts.size() );

    gc.make_best_combs_from_parts();
    // gc.display_best_combs();

    gc.makes_comb_for_cases();
    gc.write_code_for_cases();
}

/*
  Prop: pour toutes les coupes,
    * on cherche si ça correspond à un volume élémentaire
*/
int main() {
    GlobalGenCutData gcd;
    make_cuts<2>( gcd, 4 );
    // make_cuts<3>( gcd );
}

