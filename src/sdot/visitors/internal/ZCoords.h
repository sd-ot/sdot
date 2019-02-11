#pragma once

#include <cstdint>

/*
*/
template<class TZ,int dim,int nb_bits_per_axis>
struct ZCoords {
    template                <int num_axis,int _cur_bit = dim * nb_bits_per_axis - 1>
    struct                  _ZcoordsZerosOnAxis {
        static constexpr TZ v_loc = TZ( _cur_bit % dim == num_axis ? 0 : 1 ) << _cur_bit;
        static constexpr TZ value = v_loc | _ZcoordsZerosOnAxis<num_axis,_cur_bit-1>::value;
    };

    template                <int num_axis>
    struct                  _ZcoordsZerosOnAxis<num_axis,-1> {
        static constexpr TZ value = 0;
    };

    /// Ex: axis = 0, dim = 3 (i.e. x) => 000... for level and free_bits ++ 001001001...
    template                <int num_axis,int _cur_bit = dim * nb_bits_per_axis - 1>
    struct                  _ZcoordsOnesOnAxis {
        static constexpr TZ v_loc = TZ( _cur_bit % dim == num_axis ? 1 : 0 ) << _cur_bit;
        static constexpr TZ value = v_loc | _ZcoordsOnesOnAxis<num_axis,_cur_bit-1>::value;
    };
    template                <int num_axis>
    struct                  _ZcoordsOnesOnAxis<num_axis,-1> {
        static constexpr TZ value = 0;
    };
};

extern const std::uint32_t morton_256_2D_x[ 256 ];
extern const std::uint32_t morton_256_2D_y[ 256 ];
extern const std::uint32_t morton_256_3D_x[ 256 ];
extern const std::uint32_t morton_256_3D_y[ 256 ];
extern const std::uint32_t morton_256_3D_z[ 256 ];
