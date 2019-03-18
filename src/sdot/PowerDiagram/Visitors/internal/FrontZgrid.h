#pragma once

#include <ostream>
#include <vector>
#include <queue>

/***/
template<class ZG>
class FrontZgrid {
public:
    enum {                        dim = ZG::dim };
    using                         TF  = typename ZG::TF;
    using                         TI  = typename ZG::TI;
    using                         Pt  = typename ZG::Pt;

    struct                        Item {
        void                      write_to_stream( std::ostream &os ) const { os << num_grid << ":" << num_cell; }
        bool                      operator<( const Item &that ) const { return dist > that.dist; }
        TI                        num_grid;
        TI                        num_cell;
        TF                        dist;
    };

    /**/                          FrontZgrid        ( TI& op_count, std::vector<std::vector<TI>> &visited );
    template<class Grid> void     init              ( const std::vector<Grid> &grids, TI num_grid, TI num_cell, Pt position, TF weight );

    template<class Grid> void     push_without_check( TI num_grid, TI num_cell, const std::vector<Grid> &grids );
    template<class Grid> void     set_visited       ( const std::vector<Grid> &grids, TI num_grid, TI num_cell );
    bool                          empty             () const;
    template<class Cell> TF       dist              ( const Cell &cell, TF max_weight );
    template<class Grid> void     push              ( TI num_grid, TI num_cell, const std::vector<Grid> &grids );
    Item                          pop               ();

    Pt                            orig_position;
    TF                            orig_weight;
    TI&                           op_count;
    std::vector<std::vector<TI>>& visited;
    std::priority_queue<Item>     items;
};

#include "FrontZgrid.tcc"
