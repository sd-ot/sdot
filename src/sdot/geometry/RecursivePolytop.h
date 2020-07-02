#ifndef SDOT_RECURSIVE_POLYTOP_HEADER
#define SDOT_RECURSIVE_POLYTOP_HEADER

#include "internal/RecursivePolytopVertex.h"
#include "internal/RecursivePolytopImpl.h"
#include "../support/FsVec.h"
#include <deque>

/**

*/
template<class TF_,int dim_,class TI_=std::size_t,class UserData_=Void>
class RecursivePolytop {
public:
    using                   UserData        = UserData_;
    using                   Vertex          = RecursivePolytopVertex<TF_,dim_,TI_,UserData>;
    enum {                  dim             = dim_ };
    using                   TF              = TF_;
    using                   TI              = TI_;
    using                   Pt              = Point<TF,dim>;

    /**/                    RecursivePolytop( std::initializer_list<Pt> pts );
    /**/                    RecursivePolytop( const std::vector<Pt> &pts );
    /**/                    RecursivePolytop( TI nb_vertices = 0 );

    const Vertex&           vertex          ( TI i ) const { return vertices[ i ]; }
    Vertex&                 vertex          ( TI i ) { return vertices[ i ]; }

    void                    make_convex_hull();
    void                    write_to_stream ( std::ostream &os, std::string nl = "\n  ", std::string ns = "  " ) const;
    //template<class Nd> bl valid_node_prop ( const std::vector<Nd> &prop, std::vector<Pt> prev_centers = {}, bool prev_centers_are_valid = true ) const;
    template<class VO> void display_vtk     ( VO &vo ) const;
    RecursivePolytop        plane_cut       ( Pt orig, Pt normal, const std::function<UserData(const UserData &,const UserData &,TF,TF)> &nf = {} ) const;
    TF                      measure         () const;

private:
    using                   Impl            = RecursivePolytopImpl<RecursivePolytop,dim>;

    BumpPointerPool         pool;           ///< to be defined before vertices
    IntrusiveList<Impl>     impls;          ///<
    FsVec<Vertex>           vertices;       ///<
};

#include "RecursivePolytop.tcc"

#endif // SDOT_RECURSIVE_POLYTOP_HEADER
