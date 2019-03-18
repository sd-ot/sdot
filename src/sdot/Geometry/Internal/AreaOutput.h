#pragma once

#include "../../Integration/FunctionEnum.h"
#include "../../Geometry/Point3.h"

namespace sdot {

/**
*/
template<class Fu,class TF>
class AreaOutput;

/**
*/
template<class TF>
class AreaOutput<FunctionEnum::Unit,TF> {
public:
    using PT = Point3<TF>;
    using CV = int;

    AreaOutput() : area( 0 ) {}

    void add_arc    ( PT center, PT A, PT B, PT tangent, const CV &cell_value = {}, unsigned n = 50 ) {}
    void add_point  ( PT p, const CV &cell_value = {} ) {}
    void add_lines  ( const std::vector<PT> &p, const CV &cell_value = {} ) {}
    void add_arrow  ( PT center, PT dir, const CV &cell_value = {} ) {}
    void add_circle ( PT center, PT normal, TF radius, const CV &cell_value = {}, unsigned n = 50 ) {}
    void add_polygon( const std::vector<PT> &p, const CV &cell_value = {} ) {
        for( size_t i = 1; i + 1 < p.size(); ++i ) {
            PT p0 = p[ 0     ];
            PT p1 = p[ i + 0 ];
            PT p2 = p[ i + 1 ];
            area += norm_2( cross_prod( p1 - p0, p2 - p0 ) ) / 2;
        }
    }

    TF   area;
};

} // namespace sdot
