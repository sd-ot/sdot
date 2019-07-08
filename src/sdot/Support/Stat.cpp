#include <iostream>
#include <iomanip>
#include "Stream.h"
#include "Stat.h"
#include "Mpi.h"

namespace sdot {

Stat stat;

Stat::~Stat() {
    size_t max_length = 0;
    for( auto td : stats )
        max_length = std::max( max_length, td.first.size() );

    for( const auto &st : stats ) {
        if ( st.second.step ) {
            double min = + std::numeric_limits<double>::max();
            double max = - std::numeric_limits<double>::max();
            for( auto value : st.second.values ) {
                min = std::min( min, value );
                max = std::max( max, value );
            }
            std::ptrdiff_t index_min = min / st.second.step;
            std::ptrdiff_t index_max = max / st.second.step;

            std::vector<std::size_t> d( index_max - index_min + 1, 0 );
            for( auto value : st.second.values )
                d[ value / st.second.step - index_min ]++;

            std::vector<double> n( d.size(), 0 );
            for( std::size_t i = 0; i < d.size(); ++i )
                n[ i ] = double( d[ i ] ) / st.second.values.size();

            std::cout << st.first
                      << "\n    n: " << n
                      << "\n    d: " << d << "\n";
        } else {
            double sum = 0;
            for( auto value : st.second.values )
                sum += value;

            double mean = sum / st.second.values.size();
            std::cout << st.first << std::string( max_length - st.first.size(), ' ' )
                      << " -> sum: " << std::setprecision( 4 ) << std::setw( 11 ) << sum
                      << " mean: "   << std::setw( 11 ) << mean << "\n";
        }
    }
}

}
