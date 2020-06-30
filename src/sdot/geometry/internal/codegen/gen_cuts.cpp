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

    P( gc.ref_shapes );
}

