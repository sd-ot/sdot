#pragma once

#include "VtkOutput.h"
#include <string>
#include <map>

namespace sdot {

/**
*/
class ShapeType {
public:
    using                   TI            = std::size_t;

//    virtual std::vector<BI> cut_poss_count() const = 0;
//    virtual void            display_vtk   ( VtkOutput &vo, const double **tfs, const BI **tis, unsigned dim, BI nb_items, VtkOutput::Pt *offsets ) const = 0;
//    virtual void            cut_rese      ( const std::function<void(const ShapeType *,BI)> &fc, KernelSlot *ks, const ShapeData &sd ) const = 0;
    virtual unsigned        nb_nodes      () const = 0;
    virtual unsigned        nb_faces      () const = 0;
//    virtual void            cut_ops       ( KernelSlot *ks, std::map<const ShapeType *,ShapeData> &new_shape_map, const ShapeData &old_shape_data, const void *cut_ids, BI dim ) const = 0;
    virtual std::string     name          () const = 0;
};

}
