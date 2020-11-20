#pragma once

#include <parex/containers/Vec.h>
#include <parex/Value.h>

namespace sdot {

/**
*/
class PowerDiagram {
public:
    using                TaskRef      = parex::TaskRef;
    using                Value        = parex::Value;

    /**/                 PowerDiagram ( int dim );

    void                 add_diracs   ( const parex::Value &diracs ); ///< Tensor( { nb_diracs, dim + 1 } ) => positions + weights

    void                 for_each_cell( const std::function<void(const Value &elems)> &f );

private:
    TaskRef              get_min_max  ();

    std::vector<TaskRef> diracs;      ///< set of tensors with positions + weights
    int                  dim;
};

} // namespace sdot
