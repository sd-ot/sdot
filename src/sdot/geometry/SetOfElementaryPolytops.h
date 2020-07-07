#ifndef SDOT_CONVEX_POLYTOP_H
#define SDOT_CONVEX_POLYTOP_H

#include "../support/StructOfArrays.h"
#include "../support/simd/SimdVec.h"
#include "../support/PrevPow2.h"
#include "VtkOutput.h"
#include <functional>
#include <array>
#include <map>

/**
*/
template<int dim_,int nvi_=dim_,class TF_=double,class TI_=std::size_t,class Arch=CpuArch::Native>
class SetOfElementaryPolytops {
public:
    using                   TF                      = TF_;
    using                   TI                      = TI_;
    static constexpr TI     dim                     = dim_;
    static constexpr TI     nvi                     = nvi_;
    using                   Pt                      = Point<TF,dim>;

    //    /***/             ConvexPolytopSet        ( const ConvexPolytop &that );
    //    /***/             ConvexPolytopSet        ( ConvexPolytop &&that );
    /***/                   SetOfElementaryPolytops ();
    /***/                  ~SetOfElementaryPolytops ();

    // ConvexPolytop&       operator=               ( const ConvexPolytop &that );
    // ConvexPolytop&       operator=               ( ConvexPolytop &&that );

    void                    add_shape               ( const std::string &name, const std::vector<Pt> pos, TI beg_id, TI nb_elems = 1, TI face_id = 0 ); ///< names in 2D flat edges: "3", "4"... (nb points). In 3D: "3E", "4S"... => nb points in the xy cirle + Extrusion/Simple, ... Add a E or a S for each new dim
    void                    plane_cut               ( std::array<const TF *,dim> dirs, const TF *sps ); ///< cut for each elementary polytop id
    void                    clear                   ();

    void                    write_to_stream         ( std::ostream &os ) const;
    void                    display_vtk             ( VtkOutput &vo, const std::function<VtkOutput::Pt( TI id )> &offset = {} ) const;

    void                    get_measures            ( TF *measures ) const;

    //    void              add_weighted_barycenters( Pt *weighted_centers, TF *measures ) const; ///< beware: user is responsible to 0 the data
    //    void              add_measures            ( TF *measures ) const; ///< beware: user is responsible to 0 the data
    //    void              split                   ( ConvexPolytop *volumes_assemblies, bool copy_id = false ) const;

private:
    struct                  FaceIds                 { using T = std::vector<TI>; };
    struct                  Pos                     { using T = std::vector<std::array<TF,dim+1>>; }; ///< position + room for a scalar product
    struct                  Id                      { using T = TI; };

    using                   AlignedVecTI            = std::vector<TI,AlignedAllocator<TI,Arch>>;
    using                   AlignedVecTF            = std::vector<TF,AlignedAllocator<TF,Arch>>;
    using                   ShapeCoords             = StructOfArrays<std::tuple<Pos,Id,FaceIds>,Arch,TI>;
    using                   ShapeMap                = std::map<std::string,ShapeCoords>; ///< shape name => all the shapes of this type
    // using                TFCalc                  = StructOfArrays<std::vector<TF>,Arch,TI>;

    using                   CutChunkSizeCalc        = PrevPow2<32768/sizeof(TF)/(dim+1)*2>; // helper for cut_chunk_...
    enum {                  cut_chunk_size          = CutChunkSizeCalc::value }; // nb items in chunks for plane cut (to fit in L1)
    enum {                  cut_chunk_expo          = CutChunkSizeCalc::expo }; // log_2( cut_chunk_size )

    static TI               max_nb_vertices_per_elem();
    static TI               nb_faces_for            ( const std::string &name );
    static TI               nb_vertices_for         ( const std::string &name );


    template<int n> void    make_sp_and_cases       ( std::array<const TF *,dim> dirs, const TF *sps, ShapeCoords &sc, TI be, N<n>, const std::map<std::string,std::vector<TI>> &nb_created );
    void                    make_sp_and_cases       ( std::array<const TF *,dim> dirs, const TF *sps, ShapeCoords &sc, TI be, N<3>, const std::map<std::string,std::vector<TI>> &nb_created );
    //static void           reserve_and_clear       ( TFCalc &calc, TI nb_rows, TI size );
    ShapeCoords&            shape_list              ( ShapeMap &shape_map, const std::string &name, TI new_rese = 1024 );
    void                    reserve                 ( ShapeCoords &sc, TI old_size, TI new_rese );
    void                    free                    ( ShapeCoords &sc );

    AlignedVecTI            tmp_offsets_bcc;        ///< offsets in tmp_indices_bcc for each cut_case and each simd lane
    AlignedVecTI            tmp_indices_bcc;        ///<
    ShapeMap                tmp_shape_map;          ///< tmp shape_map for the cuts
    ShapeMap                shape_map;              ///< type elem => coords
    // mutable TFCalc       tf_calc;                ///< tmp storage for each elem. Typically used by get_measures, get_centroids, ...
    // std::vector<TI>      nb_inds;                ///< nb indices for each cut case, for each simd index
    TI                      end_id;                 ///< nb ids
    AlignedVecTF            tmp_f;                  ///<
};

#include "SetOfElementaryPolytops.tcc"

#endif // SDOT_CONVEX_POLYTOP_H
