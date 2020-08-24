#ifndef SDOT_SIMPLEX_SUM_POLYTOP_H
#define SDOT_SIMPLEX_SUM_POLYTOP_H

#include "../support/CpuArch.h"
#include "VtkOutput.h"
#include <functional>
#include <array>

/**
  Polytop made of a set of simplices
*/
template<int dim_,class TF_=double,class TI_=std::size_t,class Arch=CpuArch::Native>
class SimplexSumPolytop {
public:
    static constexpr TI_         dim               = dim_;
    using                        TF                = TF_;
    using                        TI                = TI_;
    using                        Pt                = Point<TF,dim>;

    /**/                         SimplexSumPolytop ( std::array<Pt,dim+1> positions ); ///<
    /**/                         SimplexSumPolytop ( Pt center, TF radius ); ///< make an simplex englobing the sphere determined by center and radius

    void                         write_to_stream   ( std::ostream &os ) const;
    void                         display_vtk       ( VtkOutput &vo ) const;
    TF                           measure           () const;

    TI                           plane_cut         ( Pt pos, Pt dir ); ///< return the cut_id

private:
    template<int nvi> struct     GenSimplex        { std::array<TI,nvi+1> nodes; std::array<TI,nvi+1> cut_ids; };
    using                        Simplex           = GenSimplex<dim>;

    template<int nvi> void       plane_cut_        ( std::vector<GenSimplex<nvi>> &new_simplices, std::vector<GenSimplex<nvi-1>> &new_faces, TI *new_nodes, const Pt pos, const Pt dir, const GenSimplex<nvi> &simplex );
    void                         plane_cut_        ( std::vector<GenSimplex<  1>> &new_simplices, std::vector<GenSimplex<    0>> &new_faces, TI *new_nodes, const Pt pos, const Pt dir, const GenSimplex<  1> &simplex );
    TF                           measure_          ( const Simplex &simplex ) const;

    std::vector<Simplex>         simplices;
    std::vector<Pt>              positions;
    TI                           nb_cuts;
};

#include "SimplexSumPolytop.tcc"

#endif // SDOT_SIMPLEX_SUM_POLYTOP_H
