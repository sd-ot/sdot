#include <fstream>
#include <random>
#include "OptParm.h"
#include "P.h"

OptParm::OptParm( std::string filename, bool random ) : random( random ) {
    count = 0;

    if ( filename.size() ) {
        std::ifstream f( filename );
        while ( true ) {
            std::size_t cur, max;
            f >> cur >> max;
            if ( ! f )
                break;
            previous_values.push_back( { cur, max } );
        }
    }
}

void OptParm::save( std::string filename ) {
    if ( filename.size() ) {
        int cpt = 0;
        std::ofstream f( filename );
        for( const Value &v : current_values )
            f << ( cpt++ ? " " : "" ) << v.val << " " << v.max;
    }
}

double OptParm::completion() const {
    if ( random )
        return 1.0 * count / random;
    double res = 0, mul = 1;
    for( const Value &v : current_values ) {
        mul /= v.max;
        res += mul * v.val;
    }
    return res;
}

std::size_t OptParm::get_value( std::size_t max, int loc_random ) {
    if ( max <= 1 )
        return 0;

    std::size_t res = 0ul;
    if ( loc_random == 0 ? random : loc_random > 0 )
        res = std::size_t( rand() ) % max;
    else if ( current_values.size() < previous_values.size() )
        res = previous_values[ current_values.size() ].val;

    current_values.push_back( { res, max } );
    return res;
}

void OptParm::restart() {
    previous_values = std::move( current_values );
}

bool OptParm::inc( bool from_current ) {
    if ( from_current )
        restart();
    if ( random )
        return ++count < random;

    while ( previous_values.size() ) {
        auto &p = previous_values.back();
        if ( ++p.val < p.max )
            return true;
        previous_values.pop_back();
    }
    return false;
}
