#include "BinStream.h"
#include <iostream>
#include <iomanip>
#include "Stat.h"
#include "Mpi.h"

Stat stat;

Stat::~Stat() {
    size_t max_length = 0;
    for( auto td : stats )
        max_length = std::max( max_length, td.first.size() );

    for( const auto &st : stats ) {
        double sum = 0;
        for( auto value : st.second.values )
            sum += value;

        double mean = sum / st.second.values.size();
        std::cout << st.first << std::string( max_length - st.first.size(), ' ' )
                  << " -> sum: " << std::setprecision( 4 ) << std::setw( 11 ) << sum
                  << " mean: "   << std::setw( 11 ) << mean << "\n";
    }
}
