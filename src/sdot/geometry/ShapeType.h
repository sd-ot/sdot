#pragma once

#include "../kernels/KernelSlot.h"
#include <string>

namespace sdot {

/**
*/
class ShapeType {
public:
    virtual unsigned    nb_nodes() const = 0;
    virtual unsigned    nb_faces() const = 0;
    virtual void        cut_ops (  ) const = 0;
    virtual std::string name    () const = 0;
};

}
