#pragma once

#include <ostream>
#include <tuple>

template<class Rp>
struct RecursivePolytopConnectivityFace {
    void write_to_stream( std::ostream &os ) const { os << ( neg ? "-" : "+" ) << item->num; }
    bool operator<      ( const RecursivePolytopConnectivityFace &that ) const { return std::tie( neg, item ) < std::tie( that.neg, that.item ); }

    Rp*  item;
    bool neg;
};
