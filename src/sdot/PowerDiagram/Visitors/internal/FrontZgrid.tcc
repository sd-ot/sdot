#include "../../system/Assert.h"
#include "FrontZgrid.h"
#include <cmath>

template<class ZG>
FrontZgrid<ZG>::FrontZgrid( TI& op_count, std::vector<std::vector<TI>> &visited ) : op_count( op_count ), visited( visited ) {
}

template<class ZG> template<class Grid>
void FrontZgrid<ZG>::init( const std::vector<Grid> &grids, TI num_grid, TI num_cell, Pt position, TF weight ) {
    ++op_count;
    set_visited( grids, num_grid, num_cell );

    orig_position = position;
    orig_weight = weight;
}

template<class ZG> template<class Grid>
void FrontZgrid<ZG>::set_visited( const std::vector<Grid> &grids, TI num_grid, TI num_cell ) {
    visited[ num_grid ][ num_cell ] = op_count;
}

template<class ZG> template<class Cell>
typename FrontZgrid<ZG>::TF FrontZgrid<ZG>::dist( const Cell &cell, TF max_weight ) {
    return norm_2_p2( cell.pos - orig_position );
}

template<class ZG> template<class Grid>
void FrontZgrid<ZG>::push_without_check( TI num_grid, TI num_cell, const std::vector<Grid> &grids ) {
    // ASSERT( visited[ num_grid ][ num_cell ] != op_count, "" );
    items.push( Item{ num_grid, num_cell, dist( grids[ num_grid ].cells[ num_cell ], grids[ num_grid ].max_weight ) } );
    set_visited( grids, num_grid, num_cell );
}

template<class ZG> template<class Grid>
void FrontZgrid<ZG>::push( TI num_grid, TI num_cell, const std::vector<Grid> &grids ) {
    if ( visited[ num_grid ][ num_cell ] != op_count )
        push_without_check( num_grid, num_cell, grids );
}

template<class ZG>
typename FrontZgrid<ZG>::Item FrontZgrid<ZG>::pop() {
    Item res = items.top();
    items.pop();
    return res;
}

template<class ZG>
bool FrontZgrid<ZG>::empty() const {
    return items.empty();
}
