#include <parex/generic_ostream_output.h>
#include <parex/TypedVariant.h>
#include <sstream>

template<class T>
TypedVariant<std::string> *to_string( TypedVariant<T> &value ) {
    std::ostringstream ss;
    ss << *value;

    return new TypedVariant<std::string>( new std::string( ss.str() ) );
}
