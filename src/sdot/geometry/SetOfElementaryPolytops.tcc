#include "../support/simd/SimdRange.h"
#include "../support/ASSERT.h"
#include "../support/range.h"
#include "../support/TODO.h"
#include "../support/P.h"

#include "SetOfElementaryPolytops.h"

namespace sdot {

#include "internal/generated/SetOfElementaryPolytopsVecOps_2D.h"

template<int dim,int nvi,class TF,class TI,class Arch>
SetOfElementaryPolytops<dim,nvi,TF,TI,Arch>::SetOfElementaryPolytops( const Arch &arch ) : end_id( 0 ), arch( arch ) {
    // tf_calc( { ( dim + 1 ) * max_nb_vertices_per_elem() } ),
}

template<int dim,int nvi,class TF,class TI,class Arch>
void SetOfElementaryPolytops<dim,nvi,TF,TI,Arch>::add_shape( const std::string &name, const std::vector<Pt> pos, TI nb_elems, TI beg_id, TI face_id ) {
    ShapeCoords &sc = shape_list( shape_map, name );

    TI old_size = sc.size();
    sc.resize( old_size + nb_elems );

    auto &s_face_ids = sc[ FaceIds() ];
    auto &s_pos = sc[ Pos() ];
    auto &s_id = sc[ Id() ];

    for( TI n = 0; n < s_pos.size(); ++n )
        for( TI d = 0; d < dim; ++d )
            s_pos[ n ][ d ].vec.fill( old_size, sc.size(), pos[ n ][ d ] );

    for( TI n = 0; n < s_face_ids.size(); ++n )
        s_face_ids[ n ].vec.fill( old_size, sc.size(), face_id );

    s_id.vec.fill_iota( old_size, sc.size(), beg_id );

    end_id = std::max( end_id, beg_id + nb_elems );
}

template<int dim,int nvi,class TF,class TI,class Arch>
typename SetOfElementaryPolytops<dim,nvi,TF,TI,Arch>::ShapeCoords& SetOfElementaryPolytops<dim,nvi,TF,TI,Arch>::shape_list( ShapeMap &shape_map, const std::string &name, TI new_rese ) {
    auto iter = shape_map.find( name );
    if ( iter == shape_map.end() ) {
        iter = shape_map.insert( iter, { name, ShapeCoords{ {
            nb_vertices_for( name ), // nb nodes for pos (and scalar product)
            nb_faces_for( name ) // face ids
        }, new_rese } } );
    }

    ShapeCoords &sl = iter->second;
    sl.reserve( new_rese );
    return sl;
}

template<int dim,int nvi,class TF,class TI,class Arch>
TI SetOfElementaryPolytops<dim,nvi,TF,TI,Arch>::nb_vertices_for( const std::string &name ) {
    #include "internal/generated/SetOfElementaryPolytops_nb_vertices_2D.h"
    PE( name );
    TODO;
    return 0;
}

template<int dim,int nvi,class TF,class TI,class Arch>
TI SetOfElementaryPolytops<dim,nvi,TF,TI,Arch>::nb_faces_for( const std::string &name ) {
    #include "internal/generated/SetOfElementaryPolytops_nb_faces_2D.h"
    PE( name );
    TODO;
    return 0;
}

template<int dim,int nvi,class TF,class TI,class Arch>
void SetOfElementaryPolytops<dim,nvi,TF,TI,Arch>::write_to_stream( std::ostream &os ) const {
    for( auto &p : shape_map )
        os << p.first << "\n" << p.second << "\n";
}

template<int dim,int nvi,class TF,class TI,class Arch>
void SetOfElementaryPolytops<dim,nvi,TF,TI,Arch>::clear() {
    for( auto &p : shape_map )
        p.second.size = 0;
    end_id = 0;
}

template<int dim,int nvi,class TF,class TI,class Arch>
TI SetOfElementaryPolytops<dim,nvi,TF,TI,Arch>::max_nb_vertices_per_elem() {
    #include "internal/generated/SetOfElementaryPolytops_max_nb_vertices_2D.h"
    TODO;
    return 0;
}

template<int dim,int nvi,class TF,class TI,class Arch>
void SetOfElementaryPolytops<dim,nvi,TF,TI,Arch>::display_vtk( VtkOutput &vo, const std::function<VtkOutput::Pt( TI id )> &offset ) const {
    for( auto &named_sc : shape_map ) {
        const ShapeCoords &sc = named_sc.second;
        std::string name = named_sc.first;
        auto &pos = sc[ Pos() ];
        auto &id = sc[ Id() ];

        auto add = [&]( std::vector<TI> inds, TI vtk_id ) {
            std::vector<VtkOutput::Pt> pts( inds.size(), 0.0 );
            for( TI num_elem = 0; num_elem < sc.size(); ++num_elem ) {
                VtkOutput::Pt o( 0.0 );
                if ( offset )
                    o += offset( id[ num_elem ] );

                for( TI i = 0; i < inds.size(); ++i ) {
                    pts[ i ] = o;
                    for( TI d = 0; d < std::min( dim * 1, TI( 3 ) ); ++d )
                        pts[ i ][ d ] += pos[ inds[ i ] ][ d ][ num_elem ];
                }
                vo.add_item( pts.data(), pts.size(), vtk_id );
            }
        };

        //
        if ( name == "3" ) { add( range<TI>( 3 ), 5 ); continue; }
        if ( name == "4" ) { add( range<TI>( 4 ), 9 ); continue; }
        // if ( name == "5" ) { add( range<TI>( 5 ), 7 ); continue; }

        PE( name );
        TODO;
    }
}

template<int dim,int nvi,class TF,class TI,class Arch>
void SetOfElementaryPolytops<dim,nvi,TF,TI,Arch>::get_measures( TF *measures ) const {
    // tmp storage
    tmp_f.reserve( SimdSize<TF,Arch>::value * end_id );
    SimdRange<SimdSize<TF,Arch>::value>::for_each( SimdSize<TF,Arch>::value * end_id, [&]( TI beg, auto simd_size ) {
        using VF = SimdVec<TF,simd_size.value,Arch>;
        VF::store_aligned( tmp_f.data() + beg, 0 );
    } );

    // for each type of element
    for( auto &named_sl : shape_map ) {
        const ShapeCoords &sc = named_sl.second;
        std::string name = named_sl.first;

        #include "internal/generated/SetOfElementaryPolytops_measure_2D.h"
        PE( name );
        TODO;
    }

    // sum of data in tmp storage
    for( TI i = 0; i < end_id; ++i ) {
        using VF = SimdVec<TF,SimdSize<TF,Arch>::value,Arch>;
        VF v = VF::load_aligned( tmp_f.data() + i * SimdSize<TF,Arch>::value );
        measures[ i ] = v.sum();
    }
}

template<int dim,int nvi,class TF,class TI,class Arch>
void SetOfElementaryPolytops<dim,nvi,TF,TI,Arch>::plane_cut( std::array<const Vec<TF,Arch> *,dim> dirs, const Vec<TF,Arch> *sps ) {
    // clear sizes in tmp shapes
    for( auto &named_sl : tmp_shape_map )
        named_sl.second.resize_wo_check( 0 );

    // cut_chunk_size
    TI len_chunk = arch.L1 / ( sizeof( TF ) * dim );

    // get room in tmp_nb_indices_bcc and tmp_indices_bcc
    TI max_nb_cut_cases = TI( 1 ) << max_nb_vertices_per_elem();
    tmp_indices_bcc.reserve( max_nb_cut_cases * len_chunk );
    tmp_offsets_bcc.reserve( max_nb_cut_cases );

    // for each type of shape
    for( auto &named_sl : shape_map ) {
        std::string name = named_sl.first;
        ShapeCoords &sc = named_sl.second;

        if ( dim == 2 && name == "3" ) {
          for( TI be = 0; be < sc.size; be += len_chunk ) {
            make_sp_and_cases( dirs, sps, sc, be, len_chunk, N<3>(), { { "3", { 1, 2, 2, 1, 2, 1, 1, 0 } } } );

            //            using RVO = RecursivePolyhedronCutVecOp_2<TF,TI,Arch,Pos,Id>;
            //            RVO::cut_l0_0_0_1_1_2_2( tmp_indices_bcc.data() + 0 * cut_chunk_size, tmp_offsets_bcc[ 0 ] - 0 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 1, 2 } );
            //            RVO::cut_l0_0_0_0_1_1_2_l0_0_0_1_2_2_2( tmp_indices_bcc.data() + 1 * cut_chunk_size, tmp_offsets_bcc[ 1 ] - 1 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 2, 1 }, sc, { 1, 0, 2 } );
            //            RVO::cut_l0_0_0_0_1_1_2_l0_0_0_1_2_2_2( tmp_indices_bcc.data() + 2 * cut_chunk_size, tmp_offsets_bcc[ 2 ] - 2 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 1, 2 } );
            //            RVO::cut_l0_0_0_0_1_0_2( tmp_indices_bcc.data() + 3 * cut_chunk_size, tmp_offsets_bcc[ 3 ] - 3 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 2, 0, 1 } );
            //            RVO::cut_l0_0_0_0_1_1_2_l0_0_0_1_2_2_2( tmp_indices_bcc.data() + 4 * cut_chunk_size, tmp_offsets_bcc[ 4 ] - 4 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 2, 1 }, sc, { 0, 2, 1 } );
            //            RVO::cut_l0_0_0_0_1_0_2( tmp_indices_bcc.data() + 5 * cut_chunk_size, tmp_offsets_bcc[ 5 ] - 5 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 1, 2, 0 } );
            //            RVO::cut_l0_0_0_0_1_0_2( tmp_indices_bcc.data() + 6 * cut_chunk_size, tmp_offsets_bcc[ 6 ] - 6 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 1, 2 } );
          };
          continue;
        }


        // generate cases
        // #include "internal/generated/SetOfElementaryPolytops_cut_cases_2D.h"

        // => elem type not found
        TODO;
    }

    //    std::swap( tmp_shape_map, shape_map );
    TODO;
}

template<int dim,int nvi,class TF,class TI,class Arch> template<int nb_nodes>
void SetOfElementaryPolytops<dim,nvi,TF,TI,Arch>::make_sp_and_cases( std::array<const Vec<TF,Arch> *,dim> dirs, const Vec<TF,Arch> *sps, ShapeCoords &sc, TI beg_chunk, TI len_chunk, N<nb_nodes>, const std::map<std::string,std::vector<TI>> &nb_created ) {
    constexpr TI nb_cases = TI( 1 ) << nb_nodes;

    // clear values in tmp_offsets_bcc
    for( TI num_case = 0; num_case < nb_cases; ++num_case )
        tmp_offsets_bcc[ num_case ] = num_case * len_chunk;

    // needed ptrs
    std::array<std::array<const TF *,dim>,nb_nodes> pos_ptrs;
    for( TI n = 0; n < nb_nodes; ++n )
        for( TI d = 0; d < dim; ++d )
            pos_ptrs[ n ][ d ] = sc[ Pos() ][ n ][ d ].vec.data();

    std::array<TF *,nb_nodes> scp_ptr;
    for( TI n = 0; n < nb_nodes; ++n )
        scp_ptr[ n ] = sc[ Pos() ][ n ][ dim ].vec.data();

    const TI *id_ptr = sc[ Id() ].vec.data();

    // get indices (num_elems) for each cases
    #ifdef __CUDACC__
    if ( OnGpu<Arch>::value ) {

    } else {
    #endif // __CUDACC__

    #ifdef __CUDACC__
    }
    #endif // __CUDACC__

    SimdRange<SimdSize<TF,Arch>::value>::for_each( sc.size(), [&]( TI beg_num_elem, auto simd_size ) {
        using VF = SimdVec<TF,simd_size.value,Arch>;
        using VI = SimdVec<TI,simd_size.value,Arch>;

        // positions
        std::array<std::array<VF,dim>,nb_nodes> pos;
        for( TI n = 0; n < nb_nodes; ++n )
            for( TI d = 0; d < dim; ++d )
                pos[ n ][ d ] = VF::load_aligned( pos_ptrs[ n ][ d ] + beg_num_elem );

        // id
        VI id = VI::load_aligned( id_ptr + beg_num_elem );

        // cut info
        std::array<VF,dim> PD;
        for( TI d = 0; d < dim; ++d )
            PD[ d ] = VF::gather( dirs[ d ], id );
        VF SD = VF::gather( sps, id );

        // scalar products
        std::array<VF,nb_nodes> scp;
        for( TI n = 0; n < nb_nodes; ++n ) {
            scp[ n ] = pos[ n ][ 0 ] * PD[ 0 ];
            for( TI d = 1; d < dim; ++d )
                scp[ n ] = scp[ n ] + pos[ n ][ d ] * PD[ d ];
        }

        // num case
        VI nc = 0;
        for( TI n = 0; n < nb_nodes; ++n )
            nc = nc + ( as_SimdVec<VI>( scp[ n ] > SD ) & TI( 1 << n ) );

        // store scalar product
        for( TI n = 0; n < nb_nodes; ++n )
            VF::store_aligned( scp_ptr[ n ] + beg_num_elem, scp[ n ] - SD );

        // store indices
        for( TI i = 0; i < simd_size.value; ++i )
            tmp_indices_bcc[ tmp_offsets_bcc[ nc[ i ] ]++ ] = beg_num_elem + i;
    } );

    // reservation
    for( auto &creation_data : nb_created ) {
        TI nb_items = 0;
        for( TI i = 0; i < nb_cases; ++i )
            nb_items += ( tmp_offsets_bcc[ i ] - i * len_chunk ) * creation_data.second[ i ];

        ShapeCoords &nc = shape_list( tmp_shape_map, creation_data.first );
        nc.reserve( nc.size() + nb_items );
    }
}

} // namespace sdot
