static std::int32_t lengthTable[ 256 ] = {
    8,7,7,6,7,6,6,5,7,6,6,5,6,5,5,4,
    7,6,6,5,6,5,5,4,6,5,5,4,5,4,4,3,
    7,6,6,5,6,5,5,4,6,5,5,4,5,4,4,3,
    6,5,5,4,5,4,4,3,5,4,4,3,4,3,3,2,
    7,6,6,5,6,5,5,4,6,5,5,4,5,4,4,3,
    6,5,5,4,5,4,4,3,5,4,4,3,4,3,3,2,
    6,5,5,4,5,4,4,3,5,4,4,3,4,3,3,2,
    5,4,4,3,4,3,3,2,4,3,3,2,3,2,2,1,
    7,6,6,5,6,5,5,4,6,5,5,4,5,4,4,3,
    6,5,5,4,5,4,4,3,5,4,4,3,4,3,3,2,
    6,5,5,4,5,4,4,3,5,4,4,3,4,3,3,2,
    5,4,4,3,4,3,3,2,4,3,3,2,3,2,2,1,
    6,5,5,4,5,4,4,3,5,4,4,3,4,3,3,2,
    5,4,4,3,4,3,3,2,4,3,3,2,3,2,2,1,
    5,4,4,3,4,3,3,2,4,3,3,2,3,2,2,1,
    4,3,3,2,3,2,2,1,3,2,2,1,2,1,1,0,
};
static std::uint8_t vecDecodeTableByte[ 256 ][ 8 ] = {
    { 0, 1, 2, 3, 4, 5, 6, 7 },
    { 1, 2, 3, 4, 5, 6, 7, 128 },
    { 0, 2, 3, 4, 5, 6, 7, 128 },
    { 2, 3, 4, 5, 6, 7, 128, 128 },
    { 0, 1, 3, 4, 5, 6, 7, 128 },
    { 1, 3, 4, 5, 6, 7, 128, 128 },
    { 0, 3, 4, 5, 6, 7, 128, 128 },
    { 3, 4, 5, 6, 7, 128, 128, 128 },
    { 0, 1, 2, 4, 5, 6, 7, 128 },
    { 1, 2, 4, 5, 6, 7, 128, 128 },
    { 0, 2, 4, 5, 6, 7, 128, 128 },
    { 2, 4, 5, 6, 7, 128, 128, 128 },
    { 0, 1, 4, 5, 6, 7, 128, 128 },
    { 1, 4, 5, 6, 7, 128, 128, 128 },
    { 0, 4, 5, 6, 7, 128, 128, 128 },
    { 4, 5, 6, 7, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 5, 6, 7, 128 },
    { 1, 2, 3, 5, 6, 7, 128, 128 },
    { 0, 2, 3, 5, 6, 7, 128, 128 },
    { 2, 3, 5, 6, 7, 128, 128, 128 },
    { 0, 1, 3, 5, 6, 7, 128, 128 },
    { 1, 3, 5, 6, 7, 128, 128, 128 },
    { 0, 3, 5, 6, 7, 128, 128, 128 },
    { 3, 5, 6, 7, 128, 128, 128, 128 },
    { 0, 1, 2, 5, 6, 7, 128, 128 },
    { 1, 2, 5, 6, 7, 128, 128, 128 },
    { 0, 2, 5, 6, 7, 128, 128, 128 },
    { 2, 5, 6, 7, 128, 128, 128, 128 },
    { 0, 1, 5, 6, 7, 128, 128, 128 },
    { 1, 5, 6, 7, 128, 128, 128, 128 },
    { 0, 5, 6, 7, 128, 128, 128, 128 },
    { 5, 6, 7, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 4, 6, 7, 128 },
    { 1, 2, 3, 4, 6, 7, 128, 128 },
    { 0, 2, 3, 4, 6, 7, 128, 128 },
    { 2, 3, 4, 6, 7, 128, 128, 128 },
    { 0, 1, 3, 4, 6, 7, 128, 128 },
    { 1, 3, 4, 6, 7, 128, 128, 128 },
    { 0, 3, 4, 6, 7, 128, 128, 128 },
    { 3, 4, 6, 7, 128, 128, 128, 128 },
    { 0, 1, 2, 4, 6, 7, 128, 128 },
    { 1, 2, 4, 6, 7, 128, 128, 128 },
    { 0, 2, 4, 6, 7, 128, 128, 128 },
    { 2, 4, 6, 7, 128, 128, 128, 128 },
    { 0, 1, 4, 6, 7, 128, 128, 128 },
    { 1, 4, 6, 7, 128, 128, 128, 128 },
    { 0, 4, 6, 7, 128, 128, 128, 128 },
    { 4, 6, 7, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 6, 7, 128, 128 },
    { 1, 2, 3, 6, 7, 128, 128, 128 },
    { 0, 2, 3, 6, 7, 128, 128, 128 },
    { 2, 3, 6, 7, 128, 128, 128, 128 },
    { 0, 1, 3, 6, 7, 128, 128, 128 },
    { 1, 3, 6, 7, 128, 128, 128, 128 },
    { 0, 3, 6, 7, 128, 128, 128, 128 },
    { 3, 6, 7, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 6, 7, 128, 128, 128 },
    { 1, 2, 6, 7, 128, 128, 128, 128 },
    { 0, 2, 6, 7, 128, 128, 128, 128 },
    { 2, 6, 7, 128, 128, 128, 128, 128 },
    { 0, 1, 6, 7, 128, 128, 128, 128 },
    { 1, 6, 7, 128, 128, 128, 128, 128 },
    { 0, 6, 7, 128, 128, 128, 128, 128 },
    { 6, 7, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 4, 5, 7, 128 },
    { 1, 2, 3, 4, 5, 7, 128, 128 },
    { 0, 2, 3, 4, 5, 7, 128, 128 },
    { 2, 3, 4, 5, 7, 128, 128, 128 },
    { 0, 1, 3, 4, 5, 7, 128, 128 },
    { 1, 3, 4, 5, 7, 128, 128, 128 },
    { 0, 3, 4, 5, 7, 128, 128, 128 },
    { 3, 4, 5, 7, 128, 128, 128, 128 },
    { 0, 1, 2, 4, 5, 7, 128, 128 },
    { 1, 2, 4, 5, 7, 128, 128, 128 },
    { 0, 2, 4, 5, 7, 128, 128, 128 },
    { 2, 4, 5, 7, 128, 128, 128, 128 },
    { 0, 1, 4, 5, 7, 128, 128, 128 },
    { 1, 4, 5, 7, 128, 128, 128, 128 },
    { 0, 4, 5, 7, 128, 128, 128, 128 },
    { 4, 5, 7, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 5, 7, 128, 128 },
    { 1, 2, 3, 5, 7, 128, 128, 128 },
    { 0, 2, 3, 5, 7, 128, 128, 128 },
    { 2, 3, 5, 7, 128, 128, 128, 128 },
    { 0, 1, 3, 5, 7, 128, 128, 128 },
    { 1, 3, 5, 7, 128, 128, 128, 128 },
    { 0, 3, 5, 7, 128, 128, 128, 128 },
    { 3, 5, 7, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 5, 7, 128, 128, 128 },
    { 1, 2, 5, 7, 128, 128, 128, 128 },
    { 0, 2, 5, 7, 128, 128, 128, 128 },
    { 2, 5, 7, 128, 128, 128, 128, 128 },
    { 0, 1, 5, 7, 128, 128, 128, 128 },
    { 1, 5, 7, 128, 128, 128, 128, 128 },
    { 0, 5, 7, 128, 128, 128, 128, 128 },
    { 5, 7, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 4, 7, 128, 128 },
    { 1, 2, 3, 4, 7, 128, 128, 128 },
    { 0, 2, 3, 4, 7, 128, 128, 128 },
    { 2, 3, 4, 7, 128, 128, 128, 128 },
    { 0, 1, 3, 4, 7, 128, 128, 128 },
    { 1, 3, 4, 7, 128, 128, 128, 128 },
    { 0, 3, 4, 7, 128, 128, 128, 128 },
    { 3, 4, 7, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 4, 7, 128, 128, 128 },
    { 1, 2, 4, 7, 128, 128, 128, 128 },
    { 0, 2, 4, 7, 128, 128, 128, 128 },
    { 2, 4, 7, 128, 128, 128, 128, 128 },
    { 0, 1, 4, 7, 128, 128, 128, 128 },
    { 1, 4, 7, 128, 128, 128, 128, 128 },
    { 0, 4, 7, 128, 128, 128, 128, 128 },
    { 4, 7, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 7, 128, 128, 128 },
    { 1, 2, 3, 7, 128, 128, 128, 128 },
    { 0, 2, 3, 7, 128, 128, 128, 128 },
    { 2, 3, 7, 128, 128, 128, 128, 128 },
    { 0, 1, 3, 7, 128, 128, 128, 128 },
    { 1, 3, 7, 128, 128, 128, 128, 128 },
    { 0, 3, 7, 128, 128, 128, 128, 128 },
    { 3, 7, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 7, 128, 128, 128, 128 },
    { 1, 2, 7, 128, 128, 128, 128, 128 },
    { 0, 2, 7, 128, 128, 128, 128, 128 },
    { 2, 7, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 7, 128, 128, 128, 128, 128 },
    { 1, 7, 128, 128, 128, 128, 128, 128 },
    { 0, 7, 128, 128, 128, 128, 128, 128 },
    { 7, 128, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 4, 5, 6, 128 },
    { 1, 2, 3, 4, 5, 6, 128, 128 },
    { 0, 2, 3, 4, 5, 6, 128, 128 },
    { 2, 3, 4, 5, 6, 128, 128, 128 },
    { 0, 1, 3, 4, 5, 6, 128, 128 },
    { 1, 3, 4, 5, 6, 128, 128, 128 },
    { 0, 3, 4, 5, 6, 128, 128, 128 },
    { 3, 4, 5, 6, 128, 128, 128, 128 },
    { 0, 1, 2, 4, 5, 6, 128, 128 },
    { 1, 2, 4, 5, 6, 128, 128, 128 },
    { 0, 2, 4, 5, 6, 128, 128, 128 },
    { 2, 4, 5, 6, 128, 128, 128, 128 },
    { 0, 1, 4, 5, 6, 128, 128, 128 },
    { 1, 4, 5, 6, 128, 128, 128, 128 },
    { 0, 4, 5, 6, 128, 128, 128, 128 },
    { 4, 5, 6, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 5, 6, 128, 128 },
    { 1, 2, 3, 5, 6, 128, 128, 128 },
    { 0, 2, 3, 5, 6, 128, 128, 128 },
    { 2, 3, 5, 6, 128, 128, 128, 128 },
    { 0, 1, 3, 5, 6, 128, 128, 128 },
    { 1, 3, 5, 6, 128, 128, 128, 128 },
    { 0, 3, 5, 6, 128, 128, 128, 128 },
    { 3, 5, 6, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 5, 6, 128, 128, 128 },
    { 1, 2, 5, 6, 128, 128, 128, 128 },
    { 0, 2, 5, 6, 128, 128, 128, 128 },
    { 2, 5, 6, 128, 128, 128, 128, 128 },
    { 0, 1, 5, 6, 128, 128, 128, 128 },
    { 1, 5, 6, 128, 128, 128, 128, 128 },
    { 0, 5, 6, 128, 128, 128, 128, 128 },
    { 5, 6, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 4, 6, 128, 128 },
    { 1, 2, 3, 4, 6, 128, 128, 128 },
    { 0, 2, 3, 4, 6, 128, 128, 128 },
    { 2, 3, 4, 6, 128, 128, 128, 128 },
    { 0, 1, 3, 4, 6, 128, 128, 128 },
    { 1, 3, 4, 6, 128, 128, 128, 128 },
    { 0, 3, 4, 6, 128, 128, 128, 128 },
    { 3, 4, 6, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 4, 6, 128, 128, 128 },
    { 1, 2, 4, 6, 128, 128, 128, 128 },
    { 0, 2, 4, 6, 128, 128, 128, 128 },
    { 2, 4, 6, 128, 128, 128, 128, 128 },
    { 0, 1, 4, 6, 128, 128, 128, 128 },
    { 1, 4, 6, 128, 128, 128, 128, 128 },
    { 0, 4, 6, 128, 128, 128, 128, 128 },
    { 4, 6, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 6, 128, 128, 128 },
    { 1, 2, 3, 6, 128, 128, 128, 128 },
    { 0, 2, 3, 6, 128, 128, 128, 128 },
    { 2, 3, 6, 128, 128, 128, 128, 128 },
    { 0, 1, 3, 6, 128, 128, 128, 128 },
    { 1, 3, 6, 128, 128, 128, 128, 128 },
    { 0, 3, 6, 128, 128, 128, 128, 128 },
    { 3, 6, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 6, 128, 128, 128, 128 },
    { 1, 2, 6, 128, 128, 128, 128, 128 },
    { 0, 2, 6, 128, 128, 128, 128, 128 },
    { 2, 6, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 6, 128, 128, 128, 128, 128 },
    { 1, 6, 128, 128, 128, 128, 128, 128 },
    { 0, 6, 128, 128, 128, 128, 128, 128 },
    { 6, 128, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 4, 5, 128, 128 },
    { 1, 2, 3, 4, 5, 128, 128, 128 },
    { 0, 2, 3, 4, 5, 128, 128, 128 },
    { 2, 3, 4, 5, 128, 128, 128, 128 },
    { 0, 1, 3, 4, 5, 128, 128, 128 },
    { 1, 3, 4, 5, 128, 128, 128, 128 },
    { 0, 3, 4, 5, 128, 128, 128, 128 },
    { 3, 4, 5, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 4, 5, 128, 128, 128 },
    { 1, 2, 4, 5, 128, 128, 128, 128 },
    { 0, 2, 4, 5, 128, 128, 128, 128 },
    { 2, 4, 5, 128, 128, 128, 128, 128 },
    { 0, 1, 4, 5, 128, 128, 128, 128 },
    { 1, 4, 5, 128, 128, 128, 128, 128 },
    { 0, 4, 5, 128, 128, 128, 128, 128 },
    { 4, 5, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 5, 128, 128, 128 },
    { 1, 2, 3, 5, 128, 128, 128, 128 },
    { 0, 2, 3, 5, 128, 128, 128, 128 },
    { 2, 3, 5, 128, 128, 128, 128, 128 },
    { 0, 1, 3, 5, 128, 128, 128, 128 },
    { 1, 3, 5, 128, 128, 128, 128, 128 },
    { 0, 3, 5, 128, 128, 128, 128, 128 },
    { 3, 5, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 5, 128, 128, 128, 128 },
    { 1, 2, 5, 128, 128, 128, 128, 128 },
    { 0, 2, 5, 128, 128, 128, 128, 128 },
    { 2, 5, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 5, 128, 128, 128, 128, 128 },
    { 1, 5, 128, 128, 128, 128, 128, 128 },
    { 0, 5, 128, 128, 128, 128, 128, 128 },
    { 5, 128, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 4, 128, 128, 128 },
    { 1, 2, 3, 4, 128, 128, 128, 128 },
    { 0, 2, 3, 4, 128, 128, 128, 128 },
    { 2, 3, 4, 128, 128, 128, 128, 128 },
    { 0, 1, 3, 4, 128, 128, 128, 128 },
    { 1, 3, 4, 128, 128, 128, 128, 128 },
    { 0, 3, 4, 128, 128, 128, 128, 128 },
    { 3, 4, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 4, 128, 128, 128, 128 },
    { 1, 2, 4, 128, 128, 128, 128, 128 },
    { 0, 2, 4, 128, 128, 128, 128, 128 },
    { 2, 4, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 4, 128, 128, 128, 128, 128 },
    { 1, 4, 128, 128, 128, 128, 128, 128 },
    { 0, 4, 128, 128, 128, 128, 128, 128 },
    { 4, 128, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 3, 128, 128, 128, 128 },
    { 1, 2, 3, 128, 128, 128, 128, 128 },
    { 0, 2, 3, 128, 128, 128, 128, 128 },
    { 2, 3, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 3, 128, 128, 128, 128, 128 },
    { 1, 3, 128, 128, 128, 128, 128, 128 },
    { 0, 3, 128, 128, 128, 128, 128, 128 },
    { 3, 128, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 2, 128, 128, 128, 128, 128 },
    { 1, 2, 128, 128, 128, 128, 128, 128 },
    { 0, 2, 128, 128, 128, 128, 128, 128 },
    { 2, 128, 128, 128, 128, 128, 128, 128 },
    { 0, 1, 128, 128, 128, 128, 128, 128 },
    { 1, 128, 128, 128, 128, 128, 128, 128 },
    { 0, 128, 128, 128, 128, 128, 128, 128 },
    { 128, 128, 128, 128, 128, 128, 128, 128 },
};
static std::uint8_t spaccTable[ 256 ][ 8 ] = {
    { 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 1, 1, 1, 1, 1, 1, 1 },
    { 0, 1, 1, 1, 1, 1, 1, 1 },
    { 1, 2, 2, 2, 2, 2, 2, 2 },
    { 0, 0, 1, 1, 1, 1, 1, 1 },
    { 1, 1, 2, 2, 2, 2, 2, 2 },
    { 0, 1, 2, 2, 2, 2, 2, 2 },
    { 1, 2, 3, 3, 3, 3, 3, 3 },
    { 0, 0, 0, 1, 1, 1, 1, 1 },
    { 1, 1, 1, 2, 2, 2, 2, 2 },
    { 0, 1, 1, 2, 2, 2, 2, 2 },
    { 1, 2, 2, 3, 3, 3, 3, 3 },
    { 0, 0, 1, 2, 2, 2, 2, 2 },
    { 1, 1, 2, 3, 3, 3, 3, 3 },
    { 0, 1, 2, 3, 3, 3, 3, 3 },
    { 1, 2, 3, 4, 4, 4, 4, 4 },
    { 0, 0, 0, 0, 1, 1, 1, 1 },
    { 1, 1, 1, 1, 2, 2, 2, 2 },
    { 0, 1, 1, 1, 2, 2, 2, 2 },
    { 1, 2, 2, 2, 3, 3, 3, 3 },
    { 0, 0, 1, 1, 2, 2, 2, 2 },
    { 1, 1, 2, 2, 3, 3, 3, 3 },
    { 0, 1, 2, 2, 3, 3, 3, 3 },
    { 1, 2, 3, 3, 4, 4, 4, 4 },
    { 0, 0, 0, 1, 2, 2, 2, 2 },
    { 1, 1, 1, 2, 3, 3, 3, 3 },
    { 0, 1, 1, 2, 3, 3, 3, 3 },
    { 1, 2, 2, 3, 4, 4, 4, 4 },
    { 0, 0, 1, 2, 3, 3, 3, 3 },
    { 1, 1, 2, 3, 4, 4, 4, 4 },
    { 0, 1, 2, 3, 4, 4, 4, 4 },
    { 1, 2, 3, 4, 5, 5, 5, 5 },
    { 0, 0, 0, 0, 0, 1, 1, 1 },
    { 1, 1, 1, 1, 1, 2, 2, 2 },
    { 0, 1, 1, 1, 1, 2, 2, 2 },
    { 1, 2, 2, 2, 2, 3, 3, 3 },
    { 0, 0, 1, 1, 1, 2, 2, 2 },
    { 1, 1, 2, 2, 2, 3, 3, 3 },
    { 0, 1, 2, 2, 2, 3, 3, 3 },
    { 1, 2, 3, 3, 3, 4, 4, 4 },
    { 0, 0, 0, 1, 1, 2, 2, 2 },
    { 1, 1, 1, 2, 2, 3, 3, 3 },
    { 0, 1, 1, 2, 2, 3, 3, 3 },
    { 1, 2, 2, 3, 3, 4, 4, 4 },
    { 0, 0, 1, 2, 2, 3, 3, 3 },
    { 1, 1, 2, 3, 3, 4, 4, 4 },
    { 0, 1, 2, 3, 3, 4, 4, 4 },
    { 1, 2, 3, 4, 4, 5, 5, 5 },
    { 0, 0, 0, 0, 1, 2, 2, 2 },
    { 1, 1, 1, 1, 2, 3, 3, 3 },
    { 0, 1, 1, 1, 2, 3, 3, 3 },
    { 1, 2, 2, 2, 3, 4, 4, 4 },
    { 0, 0, 1, 1, 2, 3, 3, 3 },
    { 1, 1, 2, 2, 3, 4, 4, 4 },
    { 0, 1, 2, 2, 3, 4, 4, 4 },
    { 1, 2, 3, 3, 4, 5, 5, 5 },
    { 0, 0, 0, 1, 2, 3, 3, 3 },
    { 1, 1, 1, 2, 3, 4, 4, 4 },
    { 0, 1, 1, 2, 3, 4, 4, 4 },
    { 1, 2, 2, 3, 4, 5, 5, 5 },
    { 0, 0, 1, 2, 3, 4, 4, 4 },
    { 1, 1, 2, 3, 4, 5, 5, 5 },
    { 0, 1, 2, 3, 4, 5, 5, 5 },
    { 1, 2, 3, 4, 5, 6, 6, 6 },
    { 0, 0, 0, 0, 0, 0, 1, 1 },
    { 1, 1, 1, 1, 1, 1, 2, 2 },
    { 0, 1, 1, 1, 1, 1, 2, 2 },
    { 1, 2, 2, 2, 2, 2, 3, 3 },
    { 0, 0, 1, 1, 1, 1, 2, 2 },
    { 1, 1, 2, 2, 2, 2, 3, 3 },
    { 0, 1, 2, 2, 2, 2, 3, 3 },
    { 1, 2, 3, 3, 3, 3, 4, 4 },
    { 0, 0, 0, 1, 1, 1, 2, 2 },
    { 1, 1, 1, 2, 2, 2, 3, 3 },
    { 0, 1, 1, 2, 2, 2, 3, 3 },
    { 1, 2, 2, 3, 3, 3, 4, 4 },
    { 0, 0, 1, 2, 2, 2, 3, 3 },
    { 1, 1, 2, 3, 3, 3, 4, 4 },
    { 0, 1, 2, 3, 3, 3, 4, 4 },
    { 1, 2, 3, 4, 4, 4, 5, 5 },
    { 0, 0, 0, 0, 1, 1, 2, 2 },
    { 1, 1, 1, 1, 2, 2, 3, 3 },
    { 0, 1, 1, 1, 2, 2, 3, 3 },
    { 1, 2, 2, 2, 3, 3, 4, 4 },
    { 0, 0, 1, 1, 2, 2, 3, 3 },
    { 1, 1, 2, 2, 3, 3, 4, 4 },
    { 0, 1, 2, 2, 3, 3, 4, 4 },
    { 1, 2, 3, 3, 4, 4, 5, 5 },
    { 0, 0, 0, 1, 2, 2, 3, 3 },
    { 1, 1, 1, 2, 3, 3, 4, 4 },
    { 0, 1, 1, 2, 3, 3, 4, 4 },
    { 1, 2, 2, 3, 4, 4, 5, 5 },
    { 0, 0, 1, 2, 3, 3, 4, 4 },
    { 1, 1, 2, 3, 4, 4, 5, 5 },
    { 0, 1, 2, 3, 4, 4, 5, 5 },
    { 1, 2, 3, 4, 5, 5, 6, 6 },
    { 0, 0, 0, 0, 0, 1, 2, 2 },
    { 1, 1, 1, 1, 1, 2, 3, 3 },
    { 0, 1, 1, 1, 1, 2, 3, 3 },
    { 1, 2, 2, 2, 2, 3, 4, 4 },
    { 0, 0, 1, 1, 1, 2, 3, 3 },
    { 1, 1, 2, 2, 2, 3, 4, 4 },
    { 0, 1, 2, 2, 2, 3, 4, 4 },
    { 1, 2, 3, 3, 3, 4, 5, 5 },
    { 0, 0, 0, 1, 1, 2, 3, 3 },
    { 1, 1, 1, 2, 2, 3, 4, 4 },
    { 0, 1, 1, 2, 2, 3, 4, 4 },
    { 1, 2, 2, 3, 3, 4, 5, 5 },
    { 0, 0, 1, 2, 2, 3, 4, 4 },
    { 1, 1, 2, 3, 3, 4, 5, 5 },
    { 0, 1, 2, 3, 3, 4, 5, 5 },
    { 1, 2, 3, 4, 4, 5, 6, 6 },
    { 0, 0, 0, 0, 1, 2, 3, 3 },
    { 1, 1, 1, 1, 2, 3, 4, 4 },
    { 0, 1, 1, 1, 2, 3, 4, 4 },
    { 1, 2, 2, 2, 3, 4, 5, 5 },
    { 0, 0, 1, 1, 2, 3, 4, 4 },
    { 1, 1, 2, 2, 3, 4, 5, 5 },
    { 0, 1, 2, 2, 3, 4, 5, 5 },
    { 1, 2, 3, 3, 4, 5, 6, 6 },
    { 0, 0, 0, 1, 2, 3, 4, 4 },
    { 1, 1, 1, 2, 3, 4, 5, 5 },
    { 0, 1, 1, 2, 3, 4, 5, 5 },
    { 1, 2, 2, 3, 4, 5, 6, 6 },
    { 0, 0, 1, 2, 3, 4, 5, 5 },
    { 1, 1, 2, 3, 4, 5, 6, 6 },
    { 0, 1, 2, 3, 4, 5, 6, 6 },
    { 1, 2, 3, 4, 5, 6, 7, 7 },
    { 0, 0, 0, 0, 0, 0, 0, 1 },
    { 1, 1, 1, 1, 1, 1, 1, 2 },
    { 0, 1, 1, 1, 1, 1, 1, 2 },
    { 1, 2, 2, 2, 2, 2, 2, 3 },
    { 0, 0, 1, 1, 1, 1, 1, 2 },
    { 1, 1, 2, 2, 2, 2, 2, 3 },
    { 0, 1, 2, 2, 2, 2, 2, 3 },
    { 1, 2, 3, 3, 3, 3, 3, 4 },
    { 0, 0, 0, 1, 1, 1, 1, 2 },
    { 1, 1, 1, 2, 2, 2, 2, 3 },
    { 0, 1, 1, 2, 2, 2, 2, 3 },
    { 1, 2, 2, 3, 3, 3, 3, 4 },
    { 0, 0, 1, 2, 2, 2, 2, 3 },
    { 1, 1, 2, 3, 3, 3, 3, 4 },
    { 0, 1, 2, 3, 3, 3, 3, 4 },
    { 1, 2, 3, 4, 4, 4, 4, 5 },
    { 0, 0, 0, 0, 1, 1, 1, 2 },
    { 1, 1, 1, 1, 2, 2, 2, 3 },
    { 0, 1, 1, 1, 2, 2, 2, 3 },
    { 1, 2, 2, 2, 3, 3, 3, 4 },
    { 0, 0, 1, 1, 2, 2, 2, 3 },
    { 1, 1, 2, 2, 3, 3, 3, 4 },
    { 0, 1, 2, 2, 3, 3, 3, 4 },
    { 1, 2, 3, 3, 4, 4, 4, 5 },
    { 0, 0, 0, 1, 2, 2, 2, 3 },
    { 1, 1, 1, 2, 3, 3, 3, 4 },
    { 0, 1, 1, 2, 3, 3, 3, 4 },
    { 1, 2, 2, 3, 4, 4, 4, 5 },
    { 0, 0, 1, 2, 3, 3, 3, 4 },
    { 1, 1, 2, 3, 4, 4, 4, 5 },
    { 0, 1, 2, 3, 4, 4, 4, 5 },
    { 1, 2, 3, 4, 5, 5, 5, 6 },
    { 0, 0, 0, 0, 0, 1, 1, 2 },
    { 1, 1, 1, 1, 1, 2, 2, 3 },
    { 0, 1, 1, 1, 1, 2, 2, 3 },
    { 1, 2, 2, 2, 2, 3, 3, 4 },
    { 0, 0, 1, 1, 1, 2, 2, 3 },
    { 1, 1, 2, 2, 2, 3, 3, 4 },
    { 0, 1, 2, 2, 2, 3, 3, 4 },
    { 1, 2, 3, 3, 3, 4, 4, 5 },
    { 0, 0, 0, 1, 1, 2, 2, 3 },
    { 1, 1, 1, 2, 2, 3, 3, 4 },
    { 0, 1, 1, 2, 2, 3, 3, 4 },
    { 1, 2, 2, 3, 3, 4, 4, 5 },
    { 0, 0, 1, 2, 2, 3, 3, 4 },
    { 1, 1, 2, 3, 3, 4, 4, 5 },
    { 0, 1, 2, 3, 3, 4, 4, 5 },
    { 1, 2, 3, 4, 4, 5, 5, 6 },
    { 0, 0, 0, 0, 1, 2, 2, 3 },
    { 1, 1, 1, 1, 2, 3, 3, 4 },
    { 0, 1, 1, 1, 2, 3, 3, 4 },
    { 1, 2, 2, 2, 3, 4, 4, 5 },
    { 0, 0, 1, 1, 2, 3, 3, 4 },
    { 1, 1, 2, 2, 3, 4, 4, 5 },
    { 0, 1, 2, 2, 3, 4, 4, 5 },
    { 1, 2, 3, 3, 4, 5, 5, 6 },
    { 0, 0, 0, 1, 2, 3, 3, 4 },
    { 1, 1, 1, 2, 3, 4, 4, 5 },
    { 0, 1, 1, 2, 3, 4, 4, 5 },
    { 1, 2, 2, 3, 4, 5, 5, 6 },
    { 0, 0, 1, 2, 3, 4, 4, 5 },
    { 1, 1, 2, 3, 4, 5, 5, 6 },
    { 0, 1, 2, 3, 4, 5, 5, 6 },
    { 1, 2, 3, 4, 5, 6, 6, 7 },
    { 0, 0, 0, 0, 0, 0, 1, 2 },
    { 1, 1, 1, 1, 1, 1, 2, 3 },
    { 0, 1, 1, 1, 1, 1, 2, 3 },
    { 1, 2, 2, 2, 2, 2, 3, 4 },
    { 0, 0, 1, 1, 1, 1, 2, 3 },
    { 1, 1, 2, 2, 2, 2, 3, 4 },
    { 0, 1, 2, 2, 2, 2, 3, 4 },
    { 1, 2, 3, 3, 3, 3, 4, 5 },
    { 0, 0, 0, 1, 1, 1, 2, 3 },
    { 1, 1, 1, 2, 2, 2, 3, 4 },
    { 0, 1, 1, 2, 2, 2, 3, 4 },
    { 1, 2, 2, 3, 3, 3, 4, 5 },
    { 0, 0, 1, 2, 2, 2, 3, 4 },
    { 1, 1, 2, 3, 3, 3, 4, 5 },
    { 0, 1, 2, 3, 3, 3, 4, 5 },
    { 1, 2, 3, 4, 4, 4, 5, 6 },
    { 0, 0, 0, 0, 1, 1, 2, 3 },
    { 1, 1, 1, 1, 2, 2, 3, 4 },
    { 0, 1, 1, 1, 2, 2, 3, 4 },
    { 1, 2, 2, 2, 3, 3, 4, 5 },
    { 0, 0, 1, 1, 2, 2, 3, 4 },
    { 1, 1, 2, 2, 3, 3, 4, 5 },
    { 0, 1, 2, 2, 3, 3, 4, 5 },
    { 1, 2, 3, 3, 4, 4, 5, 6 },
    { 0, 0, 0, 1, 2, 2, 3, 4 },
    { 1, 1, 1, 2, 3, 3, 4, 5 },
    { 0, 1, 1, 2, 3, 3, 4, 5 },
    { 1, 2, 2, 3, 4, 4, 5, 6 },
    { 0, 0, 1, 2, 3, 3, 4, 5 },
    { 1, 1, 2, 3, 4, 4, 5, 6 },
    { 0, 1, 2, 3, 4, 4, 5, 6 },
    { 1, 2, 3, 4, 5, 5, 6, 7 },
    { 0, 0, 0, 0, 0, 1, 2, 3 },
    { 1, 1, 1, 1, 1, 2, 3, 4 },
    { 0, 1, 1, 1, 1, 2, 3, 4 },
    { 1, 2, 2, 2, 2, 3, 4, 5 },
    { 0, 0, 1, 1, 1, 2, 3, 4 },
    { 1, 1, 2, 2, 2, 3, 4, 5 },
    { 0, 1, 2, 2, 2, 3, 4, 5 },
    { 1, 2, 3, 3, 3, 4, 5, 6 },
    { 0, 0, 0, 1, 1, 2, 3, 4 },
    { 1, 1, 1, 2, 2, 3, 4, 5 },
    { 0, 1, 1, 2, 2, 3, 4, 5 },
    { 1, 2, 2, 3, 3, 4, 5, 6 },
    { 0, 0, 1, 2, 2, 3, 4, 5 },
    { 1, 1, 2, 3, 3, 4, 5, 6 },
    { 0, 1, 2, 3, 3, 4, 5, 6 },
    { 1, 2, 3, 4, 4, 5, 6, 7 },
    { 0, 0, 0, 0, 1, 2, 3, 4 },
    { 1, 1, 1, 1, 2, 3, 4, 5 },
    { 0, 1, 1, 1, 2, 3, 4, 5 },
    { 1, 2, 2, 2, 3, 4, 5, 6 },
    { 0, 0, 1, 1, 2, 3, 4, 5 },
    { 1, 1, 2, 2, 3, 4, 5, 6 },
    { 0, 1, 2, 2, 3, 4, 5, 6 },
    { 1, 2, 3, 3, 4, 5, 6, 7 },
    { 0, 0, 0, 1, 2, 3, 4, 5 },
    { 1, 1, 1, 2, 3, 4, 5, 6 },
    { 0, 1, 1, 2, 3, 4, 5, 6 },
    { 1, 2, 2, 3, 4, 5, 6, 7 },
    { 0, 0, 1, 2, 3, 4, 5, 6 },
    { 1, 1, 2, 3, 4, 5, 6, 7 },
    { 0, 1, 2, 3, 4, 5, 6, 7 },
    { 1, 2, 3, 4, 5, 6, 7, 8 },
};
