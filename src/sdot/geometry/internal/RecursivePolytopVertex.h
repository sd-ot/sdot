#pragma once

#include "../Point.h"

template<class TF,int dim,class TI,class UserData>
struct  RecursivePolytopVertex {
    using           Vertex   = RecursivePolytopVertex;

    UserData        user_data;
    Point<TF,dim>   pos;
    TI              num;

    mutable Vertex *tmp_v;
    mutable TF      tmp_f;
    mutable TI      date;
};
