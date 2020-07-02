#ifndef SDOT_RECURSIVE_POLYTOP_HEADER
#define SDOT_RECURSIVE_POLYTOP_HEADER

#include "internal/RecursivePolytopVertex.h"
#include "internal/RecursivePolytopImpl.h"

/**

*/
template<class TF_,int dim_,class TI_=std::size_t,class UserData_=Void>
class RecursivePolytop {
public:
    using                         UserData            = UserData_;
    using                         Vertex              = RecursivePolytopVertex<TF_,dim_,TI_,UserData>;
    enum {                        dim                 = dim_ };
    using                         TF                  = TF_;
    using                         TI                  = TI_;
    using                         Pt                  = Point<TF,dim>;

    /**/                          RecursivePolytop    ( std::initializer_list<Pt> pts );
    /**/                          RecursivePolytop    ( const std::vector<Pt> &pts );
    /**/                          RecursivePolytop    ( TI nb_vertices = 0 );

    const Vertex&                 vertex              ( TI i ) const { return vertices[ i ]; }
    Vertex&                       vertex              ( TI i ) { return vertices[ i ]; }

    void                          make_convex_hull    ();
    void                          write_to_stream     ( std::ostream &os, std::string nl = "\n  ", std::string ns = "  " ) const;
    //template<class Nd> bl       valid_node_prop     ( const std::vector<Nd> &prop, std::vector<Pt> prev_centers = {}, bool prev_centers_are_valid = true ) const;
    template<class VO> void       display_vtk         ( VO &vo ) const;
    RecursivePolytop              plane_cut           ( Pt orig, Pt normal, const std::function<UserData(const UserData &,const UserData &,TF,TF)> &nf = {} ) const;
    bool                          contains            ( const Pt &pt ) const;
    TF                            measure             () const;

private:
    template<class R,int n>       friend              struct RecursivePolytopImpl;
    using                         Impl                = RecursivePolytopImpl<RecursivePolytop,dim>;

    template<class Rpi> void      make_tmp_connections( const IntrusiveList<Rpi> &items ) const; ///< vertex.beg and vertex.end will be used to get offets in this->tmp_connections
    template<class Rpi> TI        get_connected_graphs( const IntrusiveList<Rpi> &items ) const;
    void                          mark_connected_rec  ( const Vertex *v ) const;
    TI                            num_graph           ( const Vertex *v ) const;

    BumpPointerPool               pool;               ///< (to be defined before vertices)
    mutable TI                    date;               ///< for graph operations
    IntrusiveList<Impl>           impls;              ///< faces
    FsVec<Vertex>                 vertices;           ///<
    mutable std::vector<bool>     tmp_edges;          ///<
    mutable std::vector<Vertex *> tmp_connections;    ///<
};

#include "RecursivePolytop.tcc"

#endif // SDOT_RECURSIVE_POLYTOP_HEADER
