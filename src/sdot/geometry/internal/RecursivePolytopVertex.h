#pragma once

#include "../Point.h"

template<class TF,int dim,class TI,class UserData>
struct RecursivePolytopVertex {
    using           Vertex         = RecursivePolytopVertex;
    using           Pt             = Point<TF,dim>;

    void            write_to_stream( std::ostream &os ) const { os << num; }
    bool            outside        () const { return tmp_f > 0; }
    bool            inside         () const { return ! outside(); }

    static Pt       get_pos        ( const Vertex &v ) { return v.pos; }

    UserData        user_data;
    Pt              pos;
    TI              num;

    mutable Vertex *prev_oi;       ///< used by plane cut
    mutable Vertex *tmp_v;         ///< typically used to get new vertices correspondance
    mutable TF      tmp_f;         ///<
    mutable TI      date = 0;      ///< used for graph operations
    mutable Vertex *next;          ///< typically used to cycle through the edges
    mutable TI      beg;           ///< typically used for tmp_connections
    mutable TI      end;           ///< typically used for tmp_connections
    mutable void   *t;             ///< hum
};
