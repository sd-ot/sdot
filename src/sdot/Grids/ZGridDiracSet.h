#pragma once

#include "../support/type_config.h"

namespace sdot {

/**
*/
class ZGridDiracSet {
public:
    virtual     ~ZGridDiracSet() {}

    virtual void get_base_data( TF **coords, TF *&weights, ST *&ids ) = 0;
    virtual ST   size         () = 0;
};

}
