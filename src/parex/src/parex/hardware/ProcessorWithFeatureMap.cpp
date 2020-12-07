#include "ProcessingUnitWithFeatureMap.h"
#include <sstream>

namespace parex {

void ProcessingUnitWithFeatureMap::asimd_init( std::ostream &os, const std::string &var_name, const std::string &sp ) const {
    for( const auto &p : features )
        if ( p.second.size() )
            os << sp << var_name << ".value<" << p.first << ">() = " << p.second << ";";
}

std::string ProcessingUnitWithFeatureMap::asimd_name() const {
    std::ostringstream ss;
    ss << name() << "<" << ptr_size();
    for( const auto &p : features )
        ss << "," << p.first;
    ss << ">";
    return ss.str();
}

} // namespace parex
