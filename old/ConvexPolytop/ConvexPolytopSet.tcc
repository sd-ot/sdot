#include "../support/ASSERT.h"
#include "../support/range.h"
#include "../support/TODO.h"
#include "../support/P.h"

//// nsmake alias internal/ConvexPolytopSetVecOps.h (internal%2FConvexPolytopSetVecOps.py).h
#include "internal/ConvexPolytopSetVecOps.h"
#include "ConvexPolytopSet.h"

template<int dim,int nvi,class TF,class TI,class Arch>
ConvexPolytopSet<dim,nvi,TF,TI,Arch>::ConvexPolytopSet() : tf_calc( { ( dim + 1 ) * max_nb_vertices_per_elem() } ), end_id( 0 ) {
}

template<int dim,int nvi,class TF,class TI,class Arch>
ConvexPolytopSet<dim,nvi,TF,TI,Arch>::~ConvexPolytopSet() {
}

template<int dim,int nvi,class TF,class TI,class Arch>
void ConvexPolytopSet<dim,nvi,TF,TI,Arch>::add_shape( const std::string &name, const std::vector<Pt> pos, TI id, TI face_id ) {
    ShapeCoords &sc = shape_list( shape_map, name );
    sc.reserve( sc.size + 1 );

    auto &s_face_ids = sc[ FaceIds() ];
    auto &s_pos = sc[ Pos() ];
    auto &s_id = sc[ Id() ];

    for( TI n = 0; n < s_pos.size(); ++n )
        for( TI d = 0; d < dim; ++d )
            s_pos[ n ][ d ][ sc.size ] = pos[ n ][ d ];
    for( TI n = 0; n < s_face_ids.size(); ++n )
        s_face_ids[ n ][ sc.size ] = face_id;
    s_id[ sc.size ] = id;
    ++sc.size;

    end_id = std::max( end_id, id + 1 );
}

template<int dim,int nvi,class TF,class TI,class Arch>
typename ConvexPolytopSet<dim,nvi,TF,TI,Arch>::ShapeCoords& ConvexPolytopSet<dim,nvi,TF,TI,Arch>::shape_list( ShapeMap &shape_map, const std::string &name, TI new_rese ) {
    auto iter = shape_map.find( name );
    if ( iter == shape_map.end() ) {
        TI nv = nb_vertices_for( name );
        TI nb = nb_boundaries_for( name );
        iter = shape_map.insert( iter, { name, ShapeCoords{ {
            nv, // nb nodes for pos
            TI( 1 ) << nv, // nb cuts cases
            nv, // nb nodes for spr (scalar product)
            nb // face ids
        }, new_rese } } );
    }

    ShapeCoords &sl = iter->second;
    sl.reserve( new_rese );
    return sl;
}

template<int dim,int nvi,class TF,class TI,class Arch>
TI ConvexPolytopSet<dim,nvi,TF,TI,Arch>::nb_vertices_for( const std::string &name ) {
    #include "internal/ConvexPolytopCutGen_nb_vertices.h"
    PE( name );
    TODO;
    return 0;
}

template<int dim,int nvi,class TF,class TI,class Arch>
TI ConvexPolytopSet<dim,nvi,TF,TI,Arch>::nb_boundaries_for( const std::string &name ) {
    #include "internal/ConvexPolytopCutGen_nb_boundaries.h"
    PE( name );
    TODO;
    return 0;
}

template<int dim,int nvi,class TF,class TI,class Arch>
void ConvexPolytopSet<dim,nvi,TF,TI,Arch>::write_to_stream( std::ostream &os ) const {
    for( auto &p : shape_map )
        os << p.first << "\n" << p.second << "\n";
}

template<int dim,int nvi,class TF,class TI,class Arch> template<class Pu>
Pu ConvexPolytopSet<dim,nvi,TF,TI,Arch>::pt( const std::array<const TF *,dim> &pts, TI num, S<Pu> ) const {
    using TU = typename Pu::TF;
    Pu res = TU( 0 );
    for( TI d = 0; d < std::min( TI( dim ), TI( Pu::dim ) ); ++d )
        res[ d ] = conv( pts[ d ][ num ], S<TU>() );
    return res;
}

template<int dim,int nvi,class TF,class TI,class Arch> template<class Pu>
Pu ConvexPolytopSet<dim,nvi,TF,TI,Arch>::pt( const std::array<TF *,dim> &pts, TI num, S<Pu> ) const {
    using TU = typename Pu::TF;
    Pu res = TU( 0 );
    for( TI d = 0; d < std::min( TI( dim ), TI( Pu::dim ) ); ++d )
        res[ d ] = conv( pts[ d ][ num ], S<TU>() );
    return res;
}

template<int dim,int nvi,class TF,class TI,class Arch>
typename ConvexPolytopSet<dim,nvi,TF,TI,Arch>::Pt ConvexPolytopSet<dim,nvi,TF,TI,Arch>::pt( const std::array<const TF *,dim> &pts, TI num ) const {
    Pt res;
    for( TI d = 0; d < dim; ++d )
        res[ d ] = pts[ d ][ num ];
    return res;
}

template<int dim,int nvi,class TF,class TI,class Arch>
typename ConvexPolytopSet<dim,nvi,TF,TI,Arch>::Pt ConvexPolytopSet<dim,nvi,TF,TI,Arch>::pt( const std::array<TF *,dim> &pts, TI num ) const {
    Pt res;
    for( TI d = 0; d < dim; ++d )
        res[ d ] = pts[ d ][ num ];
    return res;
}

template<int dim,int nvi,class TF,class TI,class Arch>
void ConvexPolytopSet<dim,nvi,TF,TI,Arch>::clear() {
    for( auto &p : shape_map )
        p.second.coords.size = 0;
    end_id = 0;
}

template<int dim,int nvi,class TF,class TI,class Arch> template<int nb_nodes>
void ConvexPolytopSet<dim,nvi,TF,TI,Arch>::make_sp_and_cases( std::array<const TF *,dim> dirs, const TF *sps, ShapeCoords &sc, N<nb_nodes>, const std::map<std::string,std::vector<TI>> &nb_created ) {
    std::array<std::array<const TF *,dim>,nb_nodes> pos_ptr;
    for( TI n = 0; n < nb_nodes; ++n )
        for( TI d = 0; d < dim; ++d )
            pos_ptr[ n ][ d ] = sc[ Pos() ][ n ][ d ].data;

    std::array<TF *,nb_nodes> scp_ptr;
    for( TI n = 0; n < nb_nodes; ++n )
        scp_ptr[ n ] = sc[ Pos() ][ n ][ dim ].data;

    const TI *id_ptr = sc[ Id() ].data;

    // get indices (num_elems) for each cases
    SimdRange<SimdSize<TF,Arch>::value>::for_each( sc.size, [&]( TI beg_num_elem, auto simd_size ) {
        using VF = SimdVec<TF,simd_size.value,Arch>;
        using VI = SimdVec<TI,simd_size.value,Arch>;
        // *reinterpret_cast<int *>( 0ul ) = 2;

        // positions
        std::array<std::array<VF,dim>,nb_nodes> pos;
        for( TI n = 0; n < nb_nodes; ++n )
            for( TI d = 0; d < dim; ++d )
                pos[ n ][ d ] = VF::load_aligned( pos_ptr[ n ][ d ] + beg_num_elem );

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
        for( TI n = 0; n < simd_size.value; ++n )
            *( offset_cut_cases[ nc[ n ] ]++ ) = beg_num_elem + n;
    } );

    nb_cut_cases.resize( beg_cut_cases.size() );
    for( TI n = 0; n < nb_cut_cases.size(); ++n )
        nb_cut_cases[ n ] = offset_cut_cases[ n ] - beg_cut_cases[ n ];

    // reservation
    for( auto &creation_data : nb_created ) {
        TI nb_items = 0;
        for( TI i = 0; i < creation_data.second.size(); ++i )
            nb_items += nb_cut_cases[ i ] * creation_data.second[ i ];

        ShapeCoords &nc = shape_list( tmp_shape_map, creation_data.first );
        nc.reserve( nc.size + nb_items );
    }
}

template<int dim,int nvi,class TF,class TI,class Arch>
void ConvexPolytopSet<dim,nvi,TF,TI,Arch>::reserve_and_clear( TFCalc &calc, TI nb_rows, TI size ) {
    ASSERT( nb_rows <= calc.data.size() );
    calc.reserve( size );

    for( TI n = 0; n < nb_rows; ++n ) {
        auto &vec = calc[ n ];
        SimdRange<SimdSize<TF,Arch>::value>::for_each( size, [&]( TI beg, auto simd_size ) {
            using VF = SimdVec<TF,simd_size.value,Arch>;
            VF::store_aligned( vec.data + beg, 0 );
        } );
    }
}

template<int dim,int nvi,class TF,class TI,class Arch>
TI ConvexPolytopSet<dim,nvi,TF,TI,Arch>::max_nb_vertices_per_elem() {
    #include "internal/ConvexPolytopCutGen_max_nb_vertices.h"
}

template<int dim,int nvi,class TF,class TI,class Arch>
void ConvexPolytopSet<dim,nvi,TF,TI,Arch>::get_measures( TF *measures ) const {
    // tmp storage
    TI rese = 1;
    while ( rese < end_id )
        rese *= 2;
    reserve_and_clear( tf_calc, 1, SimdSize<TF,Arch>::value * rese );
    TF *vec = tf_calc[ 0 ].data;

    // for each type of element
    for( auto &named_sl : shape_map ) {
        const ShapeCoords &sc = named_sl.second;
        std::string name = named_sl.first;

        if ( name == "3" ) {
            const TF *p_0_0_ptr = sc[ Pos() ][ 0 ][ 0 ].data;
            const TF *p_0_1_ptr = sc[ Pos() ][ 0 ][ 1 ].data;
            const TF *p_1_0_ptr = sc[ Pos() ][ 1 ][ 0 ].data;
            const TF *p_1_1_ptr = sc[ Pos() ][ 1 ][ 1 ].data;
            const TF *p_2_0_ptr = sc[ Pos() ][ 2 ][ 0 ].data;
            const TF *p_2_1_ptr = sc[ Pos() ][ 2 ][ 1 ].data;
            const TI *id_data = sc[ Id() ].data;

            SimdRange<SimdSize<TF,Arch>::value>::for_each( sc.size, [&]( TI beg, auto simd_size ) {
                using VF = SimdVec<TF,simd_size.value,Arch>;
                using VI = SimdVec<TI,simd_size.value,Arch>;

                VI ids = VI::load_aligned( id_data + beg ) + VI::iota() * rese;
                VF old = VF::gather( vec, ids );

                VF x_0 = VF::load_aligned( p_0_0_ptr + beg );
                VF y_0 = VF::load_aligned( p_0_1_ptr + beg );
                VF x_1 = VF::load_aligned( p_1_0_ptr + beg );
                VF y_1 = VF::load_aligned( p_1_1_ptr + beg );
                VF x_2 = VF::load_aligned( p_2_0_ptr + beg );
                VF y_2 = VF::load_aligned( p_2_1_ptr + beg );

                VF dx_1 = x_1 - x_0;
                VF dy_1 = y_1 - y_0;
                VF dx_2 = x_2 - x_0;
                VF dy_2 = y_2 - y_0;

                VF res = VF( TF( 1 ) / 2 ) * ( dx_1 * dy_2 - dy_1 * dx_2 );

                VF::scatter( vec, ids, old + res );
            } );
            continue;
        }

        if ( name == "4" ) {
            const TF *p_0_0_ptr = sc[ Pos() ][ 0 ][ 0 ].data;
            const TF *p_0_1_ptr = sc[ Pos() ][ 0 ][ 1 ].data;
            const TF *p_1_0_ptr = sc[ Pos() ][ 1 ][ 0 ].data;
            const TF *p_1_1_ptr = sc[ Pos() ][ 1 ][ 1 ].data;
            const TF *p_2_0_ptr = sc[ Pos() ][ 2 ][ 0 ].data;
            const TF *p_2_1_ptr = sc[ Pos() ][ 2 ][ 1 ].data;
            const TF *p_3_0_ptr = sc[ Pos() ][ 3 ][ 0 ].data;
            const TF *p_3_1_ptr = sc[ Pos() ][ 3 ][ 1 ].data;
            const TI *id_data = sc[ Id() ].data;

            SimdRange<SimdSize<TF,Arch>::value>::for_each( sc.size, [&]( TI beg, auto simd_size ) {
                using VF = SimdVec<TF,simd_size.value,Arch>;
                using VI = SimdVec<TI,simd_size.value,Arch>;

                VI ids = VI::load_aligned( id_data + beg ) + VI::iota() * rese;
                VF old = VF::gather( vec, ids );

                VF x_0 = VF::load_aligned( p_0_0_ptr + beg );
                VF y_0 = VF::load_aligned( p_0_1_ptr + beg );
                VF x_1 = VF::load_aligned( p_1_0_ptr + beg );
                VF y_1 = VF::load_aligned( p_1_1_ptr + beg );
                VF x_2 = VF::load_aligned( p_2_0_ptr + beg );
                VF y_2 = VF::load_aligned( p_2_1_ptr + beg );
                VF x_3 = VF::load_aligned( p_3_0_ptr + beg );
                VF y_3 = VF::load_aligned( p_3_1_ptr + beg );

                VF dx_1 = x_1 - x_0;
                VF dy_1 = y_1 - y_0;
                VF dx_2 = x_2 - x_0;
                VF dy_2 = y_2 - y_0;

                VF dx_3 = x_2 - x_3;
                VF dy_3 = y_2 - y_3;
                VF dx_4 = x_1 - x_3;
                VF dy_4 = y_1 - y_3;

                VF res = VF( TF( 1 ) / 2 ) * ( dx_1 * dy_2 - dy_1 * dx_2 + dx_3 * dy_4 - dy_3 * dx_4 );

                VF::scatter( vec, ids, old + res );
            } );
            continue;
        }

        PE( name );
        TODO;
    }

    // sum if data in tmp storage
    SimdRange<SimdSize<TF,Arch>::value>::for_each( end_id, [&]( TI beg, auto simd_size ) {
        using VF = SimdVec<TF,simd_size.value,Arch>;
        VF v = VF::load_aligned( vec + beg + 0 * rese );
        for( TI n = 1; n < SimdSize<TF,Arch>::value; ++n )
            v = v + VF::load_aligned( vec + beg + n * rese );
        VF::store( measures + beg, v );
    } );
}

template<int dim,int nvi,class TF,class TI,class Arch>
void ConvexPolytopSet<dim,nvi,TF,TI,Arch>::plane_cut( std::array<const TF *,dim> dirs, const TF *sps ) {
    using CpOps = ConvexPolytopSetVecOps<TF,TI,Arch,Pos,Id>;

    // clear sizes in tmp_coords
    for( auto &named_sl : tmp_shape_map )
        named_sl.second.size = 0;

    // for each type of shape
    for( auto &named_sl : shape_map ) {
        ShapeCoords &sc = named_sl.second;
        std::string name = named_sl.first;

        // clear cut case offsets
        offset_cut_cases.resize( 1 << nb_vertices_for( name ) );
        for( TI i = 0; i < offset_cut_cases.size(); ++i )
            offset_cut_cases[ i ] = sc[ CutCase() ][ i ].data;
        beg_cut_cases = offset_cut_cases;

        // generate cases
        #include "internal/ConvexPolytopCutGen_2D.h"

        // elem type not found
        TODO;
    }

    std::swap( tmp_shape_map, shape_map );
}

template<int dim,int nvi,class TF,class TI,class Arch>
void ConvexPolytopSet<dim,nvi,TF,TI,Arch>::display_vtk( VtkOutput &vo ) const {
    for( auto &named_sc : shape_map ) {
        const ShapeCoords &sc = named_sc.second;
        std::string name = named_sc.first;
        auto &pos = sc[ Pos() ];

        auto add = [&]( std::vector<TI> inds, TI vtk_id ) {
            std::vector<VtkOutput::Pt> pts( inds.size(), 0.0 );
            for( TI num_elem = 0; num_elem < sc.size; ++num_elem ) {
                for( TI i = 0; i < inds.size(); ++i ) {
                    for( TI d = 0; d < std::min( dim * 1, TI( 3 ) ); ++d )
                        pts[ i ][ d ] = pos[ inds[ i ] ][ d ][ num_elem ];
                }
                vo.add_item( pts.data(), pts.size(), vtk_id );
            }
        };

        //
        if ( name == "3" ) { add( range<TI>( 3 ), 5 ); continue; }
        if ( name == "4" ) { add( { 0, 1, 3, 2 }, 9 ); continue; }
        // if ( name == "5" ) { add( range<TI>( 5 ), 7 ); continue; }
        PE( name );
        TODO;
    }
}



