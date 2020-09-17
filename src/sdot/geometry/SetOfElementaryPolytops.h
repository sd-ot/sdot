#ifndef SDOT_SetOfElementaryPolytops_H
#define SDOT_SetOfElementaryPolytops_H

#include "../support/StructOfArrays.h"
#include "../support/simd/SimdVec.h"
#include "../support/PrevPow2.h"
#include "../support/Vec.h"
#include "VtkOutput.h"
#include <functional>
#include <array>
#include <map>

namespace sdot {

/**
*/
template<int dim_,int nvi_=dim_,class TF_=double,class TI_=std::size_t,class Arch=MachineArch::Native>
class SetOfElementaryPolytops {
public:
    using                   TF                      = TF_;
    using                   TI                      = TI_;
    static constexpr TI     dim                     = dim_;
    static constexpr TI     nvi                     = nvi_;
    using                   Pt                      = Point<TF,dim>;

    /***/                   SetOfElementaryPolytops ( const Arch &arch = {} );

    void                    add_shape               ( const std::string &name, const std::vector<Pt> pos, TI nb_elems, TI beg_id = 0, TI face_id = 0 ); ///< names in 2D flat edges: "3", "4"... (nb points). In 3D: "3E", "4S"... => nb points in the xy cirle + Extrusion/Simple, ... Add a E or a S for each new dim
    void                    plane_cut               ( std::array<const TF *,dim> cut_dirs, const TF *cut_sps ); ///< cut for each elementary polytop id
    void                    clear                   ();

    void                    write_to_stream         ( std::ostream &os ) const;
    void                    display_vtk             ( VtkOutput &vo, const std::function<VtkOutput::Pt( TI id )> &offset = {} ) const;

    void                    get_measures            ( TF *measures ) const;

    //    void              add_weighted_barycenters( Pt *weighted_centers, TF *measures ) const; ///< beware: user is responsible to 0 the data
    //    void              split                   ( ConvexPolytop *volumes_assemblies, bool copy_id = false ) const;

private:
    // virtual struct that will be transformed by StructOfArrays (std::vectors are assumed to be of the same size inside the same batch)
    struct                  FaceIds                 { using T = std::vector<TI>; };
    struct                  Pos                     { using T = std::vector<std::array<TF,dim>>; }; ///< position
    struct                  Id                      { using T = TI; };

    // types for actuak data
    using                   ShapeCoords             = StructOfArrays<std::tuple<Pos,Id,FaceIds>,Arch,TI>;
    using                   ShapeMap                = std::map<std::string,ShapeCoords>; ///< shape name => all the shapes of this type

    static TI               max_nb_vertices_per_elem();
    static TI               nb_vertices_for         ( const std::string &name );
    static TI               nb_faces_for            ( const std::string &name );


    template<int n> void    make_sp_and_cases       ( Vec<TI,Arch> &offsets, Vec<TI,Arch> &indices, Vec<TF,Arch> *sps, std::array<const TF *,dim> cut_dirs, const TF *cut_sps, TI beg_chunk, TI len_chunk, ShapeCoords &sc, N<n>, N<0> on_gpu );
    template<int n> void    make_sp_and_cases       ( Vec<TI,Arch> &offsets, Vec<TI,Arch> &indices, Vec<TF,Arch> *sps, std::array<const TF *,dim> cut_dirs, const TF *cut_sps, TI beg_chunk, TI len_chunk, ShapeCoords &sc, N<n>, N<1> on_gpu );
    ShapeCoords&            shape_list              ( ShapeMap &shape_map, const std::string &name, TI new_rese = 1024 );
    void                    reserve                 ( ShapeCoords &sc, TI old_size, TI new_rese );
    void                    free                    ( ShapeCoords &sc );

    ShapeMap                tmp_shape_map;          ///< tmp shape_map for the cuts
    ShapeMap                shape_map;              ///< type elem => coords
    TI                      end_id;                 ///< nb ids
    Arch                    arch;                   ///<
};

} // namespace sdot

#include "SetOfElementaryPolytops.tcc"

#endif // SDOT_SetOfElementaryPolytops_H
