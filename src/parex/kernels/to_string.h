#include <parex/support/generic_ostream_output.h>
#include <sstream>

template<class T>
std::string *to_string( const T &value ) {
    std::ostringstream os;
    os << value;

    return new std::string( os.str() + "cmsdo" );
}

