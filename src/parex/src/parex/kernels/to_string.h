#include <parex/generic_ostream_output.h>
#include <parex/TaskOut.h>
#include <parex/P.h>
#include <sstream>

template<class T>
TaskOut<std::string> to_string( const TaskOut<T> &value ) {
    std::ostringstream ss;
    ss << *value;

    return new std::string( ss.str() );
}
