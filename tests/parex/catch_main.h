#include <parex/utility/generic_ostream_output.h>
#include <catch2/catch_test_macros.hpp>
#include <sstream>

template<class A,class B>
bool same_repr( const A &a, const B &b ) {
    std::ostringstream oa, ob;
    oa << a;
    ob << b;
    return oa.str() == ob.str();
}