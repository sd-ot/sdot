#pragma once

namespace sdot {

/**
*/
template<class T,class S>
class ZGridDiracSet {
public:
    virtual     ~ZGridDiracSet() {}

    virtual void get_base_data( T **coords, T *&weights, S *&ids ) = 0;
    virtual S    size         () = 0;
};

}
