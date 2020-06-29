#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/rational.hpp>
#include "Conv.h"

using Rational = boost::rational<boost::multiprecision::cpp_int>;


template<class G>
G conv( const Rational &val, S<G> ) {
    return boost::rational_cast<G>( val );
}

