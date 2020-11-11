#include <parex/containers/Tensor.h>
#include <parex/containers/Vec.h>
#include <sstream>
#include <string>
#include <map>
using namespace parex;

template<class TI>
void insert_( std::map<std::string,TI> & ) {
}

template<class TI,class ...Args>
void insert_( std::map<std::string,TI> &res, const std::map<std::string,TI> &head, const Args &...tail ) {
    for( const auto &p : head ) {
        auto iter = res.find( p.first );
        if ( iter == res.end() )
            iter = res.insert( iter, { p.first, 0 } );
        iter->second += p.second;
    }

    insert_( res, tail... );
}

template<class TI,class ...Args>
std::map<std::string,TI> *plane_cut_reservation_new_elements( S<TI>, const Args &...args ) {
    std::map<std::string,TI> *res = new std::map<std::string,TI>;
    insert_( *res, args... );
    return res;
}
