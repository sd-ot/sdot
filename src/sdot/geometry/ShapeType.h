#pragma once

#include <parex/containers/Vec.h>
#include <functional>
#include <string>
#include <map>

namespace sdot {

/**
*/
class ShapeType {
public:
    using                   TI            = std::size_t;
    using                   CRN           = std::map<ShapeType *,parex::Vec<TI>>;
    struct                  OutCutOp      { ShapeType *shape_type; parex::Vec<TI> node_corr, face_corr; };
    struct                  CutOp         { std::string operation_name; parex::Vec<OutCutOp> outputs; parex::Vec<TI> inp_node_corr, inp_face_corr; TI num_case, num_sub_case; };
    using                   VecCutOp      = parex::Vec<CutOp>;

    static std::string      type_name     () { return "sdot::ShapeType"; }

    virtual parex::Vec<TI> *cut_poss_count() const = 0;
    virtual CRN            *cut_rese_new  () const = 0;
    virtual void            display_vtk   ( const std::function<void( TI vtk_id, const parex::Vec<TI> &nodes )> &f ) const = 0;
    virtual unsigned        nb_nodes      () const = 0;
    virtual unsigned        nb_faces      () const = 0;
    virtual VecCutOp       *cut_ops       () const = 0;
    virtual std::string     name          () const = 0;
};

}
