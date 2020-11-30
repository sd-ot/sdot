#include <parex/generic_ostream_output.h>
#include <parex/TaskOut.h>
#include <parex/P.h>
#include <sstream>

template<class ShapeMap,class TI,class A,class B,class C>
TaskOut<ShapeMap> add_repeated( TaskOut<ShapeMap> &shape_map, TaskOut<std::string> &shape_name, TaskOut<TI> count, TaskOut<A> &coordinates, TaskOut<B> &face_ids, TaskOut<C> &beg_ids ) {
    auto *hl = shape_map[ shape_name ];

}
