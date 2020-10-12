#pragma once

#include "../kernels/KernelSlot.h"
#include <string>

namespace sdot {
class VtkOutput;

/**
*/
class ShapeType {
public:
    using               BI         = std::uint64_t;

    virtual void        display_vtk( VtkOutput &vo, const double **tfs, const BI **tis, unsigned dim, BI nb_items ) = 0;
    virtual unsigned    nb_nodes   () const = 0;
    virtual unsigned    nb_faces   () const = 0;
    virtual void        cut_ops    (  ) const = 0;
    virtual std::string name       () const = 0;
};

}
