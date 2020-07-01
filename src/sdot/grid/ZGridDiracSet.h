#pragma once

namespace sdot {

/**
*/
template<class T,class S>
class ZGridDiracSet {
public:
    virtual     ~ZGridDiracSet  () {}

    virtual void write_to_stream( std::ostream &os, const std::string &sp = {} ) const = 0;
    virtual void get_base_data  ( T **coords, T *&weights, S *&ids ) = 0;
    virtual void add_dirac      ( const T *coords, T weight, S id ) = 0;
    virtual S    size           () = 0;
};

}