#include "Triangle.h"

namespace sdot {

///
class Triangle : public ShapeType {
public:
    virtual unsigned    nb_nodes() const override { return 3; }
    virtual unsigned    nb_faces() const override { return 3; }
    virtual void        cut_ops () const override {}
    virtual std::string name    () const override { return "triangle"; }
};

ShapeType *triangle() {
    static Triangle res;
    return &res;
}

}

