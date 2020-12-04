#include <parex/tasks/TaskOut.h>

template<class T>
parex::TaskOut<std::uint64_t> size( const parex::TaskOut<T> &value ) {
    return new std::uint64_t( value->size() );
}
