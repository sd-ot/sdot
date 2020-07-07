#ifndef SDOT_CONVEX_POLYTOP_H
#define SDOT_CONVEX_POLYTOP_H

#include "../support/StructOfArrays.h"
#include "../support/simd/SimdVec.h"
#include "../support/VtkOutput.h"
#include <functional>
#include <array>
#include <map>

/**
*/
template<int dim_,int nvi_=dim_,class TF_=double,class TI_=std::size_t,class Arch=CpuArch::Native>
class ConvexPolytopSet {
public:
    using                   TF                      = TF_;
    using                   TI                      = TI_;
    static constexpr TI     dim                     = dim_;
    static constexpr TI     nvi                     = nvi_;
    using                   Pt                      = Point<TF,dim>;

    //    /***/             ConvexPolytopSet        ( const ConvexPolytop &that );
    //    /***/             ConvexPolytopSet        ( ConvexPolytop &&that );
    /***/                   ConvexPolytopSet        ();
    /***/                  ~ConvexPolytopSet        ();

    // ConvexPolytop&       operator=               ( const ConvexPolytop &that );
    // ConvexPolytop&       operator=               ( ConvexPolytop &&that );

    void                    add_shape               ( const std::string &name, const std::vector<Pt> pos, TI id, TI face_id = 0 );
    void                    plane_cut               ( std::array<const TF *,dim> dirs, const TF *sps ); ///< cut for each id
    void                    clear                   ();

    void                    write_to_stream         ( std::ostream &os ) const;
    void                    display_vtk             ( VtkOutput &vo ) const;

    void                    get_measures            ( TF *measures ) const;

    //    void              add_weighted_barycenters( Pt *weighted_centers, TF *measures ) const; ///< beware: user is responsible to 0 the data
    //    void              add_measures            ( TF *measures ) const; ///< beware: user is responsible to 0 the data
    //    void              split                   ( ConvexPolytop *volumes_assemblies, bool copy_id = false ) const;

private:
    struct                  CutCase                 { using T = std::vector<TI>; };
    struct                  FaceIds                 { using T = std::vector<TI>; };
    struct                  Pos                     { using T = std::vector<std::array<TF,dim+1>>; }; ///< position + room for a scalar product
    struct                  Id                      { using T = TI; };

    using                   ShapeCoords             = StructOfArrays<std::tuple<Pos,Id,CutCase,FaceIds>,Arch,TI>;
    using                   ShapeMap                = std::map<std::string,ShapeCoords>; ///< shape name => all the shapes of this type
    using                   CutCases                = StructOfArrays<std::vector<TI>,Arch,TI>;
    using                   TFCalc                  = StructOfArrays<std::vector<TF>,Arch,TI>;

    static TI               max_nb_vertices_per_elem();
    static TI               nb_boundaries_for       ( const std::string &name );
    static TI               nb_vertices_for         ( const std::string &name );


    template<int n> void    make_sp_and_cases       ( std::array<const TF *,dim> dirs, const TF *sps, ShapeCoords &sc, N<n>, const std::map<std::string,std::vector<TI>> &nb_created );
    static void             reserve_and_clear       ( TFCalc &calc, TI nb_rows, TI size );
    ShapeCoords&            shape_list              ( ShapeMap &shape_map, const std::string &name, TI new_rese = 1024 );
    void                    reserve                 ( ShapeCoords &sc, TI old_size, TI new_rese );
    void                    free                    ( ShapeCoords &sc );

    template<class Pu> Pu   pt                      ( const std::array<const TF *,dim> &pts, TI num, S<Pu> ) const;
    template<class Pu> Pu   pt                      ( const std::array<TF *,dim> &pts, TI num, S<Pu> ) const;
    Pt                      pt                      ( const std::array<const TF *,dim> &pts, TI num ) const;
    Pt                      pt                      ( const std::array<TF *,dim> &pts, TI num ) const;

    std::vector<TI *>       offset_cut_cases;       ///<
    std::vector<TI *>       beg_cut_cases;          ///<
    std::vector<TI>         nb_cut_cases;           ///<
    ShapeMap                tmp_shape_map;          ///< tmp shape_map for the cuts
    ShapeMap                shape_map;              ///< type elem => coords
    mutable TFCalc          tf_calc;                ///<
    TI                      end_id;                 ///<
};

#include "ConvexPolytopSet.tcc"

#endif // SDOT_CONVEX_POLYTOP_H
