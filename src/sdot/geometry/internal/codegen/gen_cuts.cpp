#include "GenCuts.h"

/*
  Prop: pour toutes les coupes,
    * on cherche si ça correspond à un volume élémentaire
*/
int main() {
    GlobalGenCutData gcd;
    GenCuts<2> gc( gcd );
    gc.add_ref_shape( "3" );
    gc.add_ref_shape( "4" );

    for( const auto &rs : gc.ref_shapes ) {
        gc.setup_cut_nodes_for( rs );
        gc.setup_parts_from_cut_nodes();

    }
}

