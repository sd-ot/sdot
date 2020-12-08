#include <parex/tasks/TaskOut.h>
using namespace parex;

template<class O,class T>
TaskOut<O> conv_to( const TaskOut<O> &, const TaskOut<T> &value ) {
    return new O( *value );
}
