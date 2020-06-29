#include "../src/sdot/support/StructOfArrays.h"
#include "../src/sdot/support/P.h"


int main() {
    struct Pos { using T = std::vector<std::array<float,3>>; };
    struct Id { using T = int; };

    StructOfArrays<std::tuple<Pos,Id>> s( { 2 } );

    Pos pos;
    Id id;
    s.size = 1;
    s[ id ][ 0 ] = 17;
    s[ pos ][ 0 ][ 0 ][ 0 ] = 1;
    s[ pos ][ 0 ][ 1 ][ 0 ] = 2;
    s[ pos ][ 0 ][ 2 ][ 0 ] = 3;
    s[ pos ][ 1 ][ 0 ][ 0 ] = 4;
    s[ pos ][ 1 ][ 1 ][ 0 ] = 5;
    s[ pos ][ 1 ][ 2 ][ 0 ] = 6;

    P( s );
}

