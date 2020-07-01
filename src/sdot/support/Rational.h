#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/rational.hpp>
#include "conv.h"

using Rational = boost::rational<boost::multiprecision::cpp_int>;


template<class G>
G conv( const Rational &val, S<G> ) {
    return boost::rational_cast<G>( val );
}


std::ostream &operator<<( std::ostream &os, const Rational &r ) {
    os << r.numerator();
    if ( r.denominator() != 1 )
        os << "/" << r.denominator();
    return os;
}

Rational abs( const Rational &a ) {
    return a >= 0 ? a : -a;
}
