#include <parex/utility/generic_ostream_output.h>
#include <parex/tasks/TaskOut.h>
#include <sstream>

template<class T>
parex::TaskOut<std::string> to_string( const parex::TaskOut<T> &value ) {
    std::ostringstream ss;
    ss << *value;

    return new std::string( ss.str() );
}
