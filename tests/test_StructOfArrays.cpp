#include "../src/sdot/support/StructOfArrays.h"
#include "../src/sdot/support/P.h"

int main() {
    struct Pos { using T = std::vector<std::array<float,3>>; };
    struct Id { using T = int; };

    StructOfArrays<std::tuple<Pos,Id>> s( { 2 }, 2 );
    s.resize( 2 );

    Pos pos;
    Id id;

    s[ pos ][ 0 ][ 0 ][ 0 ] = 1;
    s[ pos ][ 0 ][ 1 ][ 0 ] = 2;
    s[ pos ][ 0 ][ 2 ][ 0 ] = 3;
    s[ pos ][ 1 ][ 0 ][ 0 ] = 4;
    s[ pos ][ 1 ][ 1 ][ 0 ] = 5;
    s[ pos ][ 1 ][ 2 ][ 0 ] = 6;
    s[ id ][ 0 ] = 17;

    s[ pos ][ 0 ][ 0 ][ 1 ] = 11;
    s[ pos ][ 0 ][ 1 ][ 1 ] = 12;
    s[ pos ][ 0 ][ 2 ][ 1 ] = 13;
    s[ pos ][ 1 ][ 0 ][ 1 ] = 14;
    s[ pos ][ 1 ][ 1 ][ 1 ] = 15;
    s[ pos ][ 1 ][ 2 ][ 1 ] = 16;
    s[ id ][ 1 ] = 18;

    P( s );
}

