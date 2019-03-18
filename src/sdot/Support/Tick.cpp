#include "BinStream.h"
#include <iostream>
#include <iomanip>
#include "Tick.h"
#include "Mpi.h"

Tick tick;

Tick::~Tick() {
    if ( mpi->size() > 1 ) {
        if ( mpi->rank() ) {
            Hpipe::CbQueue cq;
            Hpipe::BinStream<Hpipe::CbQueue> bq( &cq );

            bq.write_unsigned( tasks.size() );
            for( auto td : tasks ) {
                bq << td.first;
                bq << std::vector<double>( td.second.timings.begin(), td.second.timings.end() );
            }

            std::vector<std::int8_t> msg( cq.size() );
            cq.read_some( msg.data(), msg.size() );
            mpi->send( msg.data(), msg.size(), 0, mpi->rank() );
        } else {
            for( int rank = 1; rank < mpi->size(); ++rank ) {
                std::vector<std::int8_t> str( mpi->probe_size( rank, rank ) );
                mpi->recv( str.data(), str.size(), rank, rank );

                Hpipe::CmString cm( str.data(), str.data() + str.size() );
                Hpipe::BinStream<Hpipe::CmString> bq( &cm );
                for( int i = 0, n = bq.read_unsigned(); i < n; ++i ) {
                    std::string id = bq.read();
                    std::vector<double> timings = bq.read();

                    TaskData &td = tasks[ id ];
                    for( double t : timings )
                        td.timings.emplace_back( t );
                }
            }
        }
    }

    if ( mpi->rank() == 0 ) {
        auto mod_name = []( const std::string &str ) {
            size_t count = 0, last = 0;
            for( size_t i = 0; i < str.size(); ++i ) {
                if ( str[ i ] == '.' ) {
                    last = i;
                    ++count;
                }
            }
            return count ? std::string( 2 * count, ' ' ) + str.substr( last + 1 ) : str;
        };
        auto prefix = []( const std::string &str ) {
            for( int i = str.size(); i--; )
                if ( str[ i ] == '.' )
                    return str.substr( 0, i );
            return std::string{};
        };
        auto pr = []( const std::string &str ) {
            std::string res;
            for( int i = str.size(); i--; )
                if ( str[ i ] == '.' )
                    res += "  ";
            return res;
        };

        size_t max_length = 0;
        for( auto td : tasks )
            max_length = std::max( max_length, mod_name( td.first ).size() );

        for( auto td : tasks ) {
            double sum = 0;
            for( auto tp : td.second.timings )
                sum += tp;

            double total = 0.0;
            for( auto te : tasks )
                if ( prefix( td.first ) == prefix( te.first ) )
                    for( auto tp : te.second.timings )
                        total += tp;

            double mean = sum / td.second.timings.size();
            std::cout << mod_name( td.first ) << std::string( max_length - mod_name( td.first ).size(), ' ' )
                      << " -> sum: " << std::setprecision( 4 ) << std::setw( 11 ) << sum
                      << " mean: "   << std::setw( 11 ) << mean
                      << " ratio: "  << pr( td.first ) << std::setprecision( 2 ) << 100 * sum / total << "\n";
        }
    }
}
