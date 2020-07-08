template<class TF,class TI,class Arch,class Pos,class Id>
struct RecursivePolyhedronCutVecOp_2 {
    template<class ShapeCoords>
    static void cut_l0_0_0_0_1_0_2( const TI *indices_data, TI indices_size, ShapeCoords &nc_0, std::array<TI,3> dst_0_indices, const ShapeCoords &sc, std::array<TI,3> src_indices ) {
        Pos pos;
        Id id;

        const TF *src_pos_0_0_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 0 ].data;
        const TF *src_pos_0_0_1_ptr = sc[ pos ][ src_indices[ 0 ] ][ 1 ].data;
        const TF *src_pos_1_1_0_ptr = sc[ pos ][ src_indices[ 1 ] ][ 0 ].data;
        const TF *src_pos_1_1_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 1 ].data;
        const TF *src_pos_2_2_0_ptr = sc[ pos ][ src_indices[ 2 ] ][ 0 ].data;
        const TF *src_pos_2_2_1_ptr = sc[ pos ][ src_indices[ 2 ] ][ 1 ].data;

        const TI *src_id_ptr = sc[ id ].data;

        const TF *src_sp_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 2 ].data;
        const TF *src_sp_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 2 ].data;
        const TF *src_sp_2_ptr = sc[ pos ][ src_indices[ 2 ] ][ 2 ].data;

        TF *dst_0_pos_0_0_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_0_1_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_1_0_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_1_1_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_2_0_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_2_1_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 1 ].data + nc_0.size;

        TI *dst_0_id_ptr = nc_0[ id ].data + nc_0.size;

        SimdRange<SimdSize<TF,Arch>::value>::for_each( indices_size, [&]( TI beg_num_ind, auto simd_size ) {
            using VI = SimdVec<TI,simd_size.value,Arch>;
            using VF = SimdVec<TF,simd_size.value,Arch>;

            VI inds = VI::load_aligned( indices_data + beg_num_ind );

            VF src_sp_0 = VF::gather( src_sp_0_ptr, inds );
            VF src_sp_1 = VF::gather( src_sp_1_ptr, inds );
            VF src_sp_2 = VF::gather( src_sp_2_ptr, inds );

            VF src_pos_0_0_0 = VF::gather( src_pos_0_0_0_ptr, inds );
            VF src_pos_0_0_1 = VF::gather( src_pos_0_0_1_ptr, inds );
            VF src_pos_1_1_0 = VF::gather( src_pos_1_1_0_ptr, inds );
            VF src_pos_1_1_1 = VF::gather( src_pos_1_1_1_ptr, inds );
            VF src_pos_2_2_0 = VF::gather( src_pos_2_2_0_ptr, inds );
            VF src_pos_2_2_1 = VF::gather( src_pos_2_2_1_ptr, inds );

            VF m_0_1 = src_sp_0 / ( src_sp_0 - src_sp_1 );
            VF m_0_2 = src_sp_0 / ( src_sp_0 - src_sp_2 );

            VF src_pos_0_1_0 = src_pos_0_0_0 + m_0_1 * ( src_pos_1_1_0 - src_pos_0_0_0 );
            VF src_pos_0_1_1 = src_pos_0_0_1 + m_0_1 * ( src_pos_1_1_1 - src_pos_0_0_1 );
            VF src_pos_0_2_0 = src_pos_0_0_0 + m_0_2 * ( src_pos_2_2_0 - src_pos_0_0_0 );
            VF src_pos_0_2_1 = src_pos_0_0_1 + m_0_2 * ( src_pos_2_2_1 - src_pos_0_0_1 );

            VI ids = VI::gather( src_id_ptr, inds );

            VF::store( dst_0_pos_0_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_0 );
            VF::store( dst_0_pos_0_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_1 );
            VF::store( dst_0_pos_1_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_1_0 );
            VF::store( dst_0_pos_1_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_1_1 );
            VF::store( dst_0_pos_2_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_2_0 );
            VF::store( dst_0_pos_2_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_2_1 );

            VI::store( dst_0_id_ptr + 1 * beg_num_ind + 0 * simd_size.value, ids );
        } );

        nc_0.size += indices_size * 1;
    }
    template<class ShapeCoords>
    static void cut_l0_0_0_0_1_1_2_2_2( const TI *indices_data, TI indices_size, ShapeCoords &nc_0, std::array<TI,4> dst_0_indices, const ShapeCoords &sc, std::array<TI,3> src_indices ) {
        Pos pos;
        Id id;

        const TF *src_pos_0_0_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 0 ].data;
        const TF *src_pos_0_0_1_ptr = sc[ pos ][ src_indices[ 0 ] ][ 1 ].data;
        const TF *src_pos_1_1_0_ptr = sc[ pos ][ src_indices[ 1 ] ][ 0 ].data;
        const TF *src_pos_1_1_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 1 ].data;
        const TF *src_pos_2_2_0_ptr = sc[ pos ][ src_indices[ 2 ] ][ 0 ].data;
        const TF *src_pos_2_2_1_ptr = sc[ pos ][ src_indices[ 2 ] ][ 1 ].data;

        const TI *src_id_ptr = sc[ id ].data;

        const TF *src_sp_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 2 ].data;
        const TF *src_sp_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 2 ].data;
        const TF *src_sp_2_ptr = sc[ pos ][ src_indices[ 2 ] ][ 2 ].data;

        TF *dst_0_pos_0_0_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_0_1_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_1_0_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_1_1_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_2_0_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_2_1_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_3_0_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_3_1_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 1 ].data + nc_0.size;

        TI *dst_0_id_ptr = nc_0[ id ].data + nc_0.size;

        SimdRange<SimdSize<TF,Arch>::value>::for_each( indices_size, [&]( TI beg_num_ind, auto simd_size ) {
            using VI = SimdVec<TI,simd_size.value,Arch>;
            using VF = SimdVec<TF,simd_size.value,Arch>;

            VI inds = VI::load_aligned( indices_data + beg_num_ind );

            VF src_sp_0 = VF::gather( src_sp_0_ptr, inds );
            VF src_sp_1 = VF::gather( src_sp_1_ptr, inds );
            VF src_sp_2 = VF::gather( src_sp_2_ptr, inds );

            VF src_pos_0_0_0 = VF::gather( src_pos_0_0_0_ptr, inds );
            VF src_pos_0_0_1 = VF::gather( src_pos_0_0_1_ptr, inds );
            VF src_pos_1_1_0 = VF::gather( src_pos_1_1_0_ptr, inds );
            VF src_pos_1_1_1 = VF::gather( src_pos_1_1_1_ptr, inds );
            VF src_pos_2_2_0 = VF::gather( src_pos_2_2_0_ptr, inds );
            VF src_pos_2_2_1 = VF::gather( src_pos_2_2_1_ptr, inds );

            VF m_0_1 = src_sp_0 / ( src_sp_0 - src_sp_1 );
            VF m_1_2 = src_sp_1 / ( src_sp_1 - src_sp_2 );

            VF src_pos_0_1_0 = src_pos_0_0_0 + m_0_1 * ( src_pos_1_1_0 - src_pos_0_0_0 );
            VF src_pos_0_1_1 = src_pos_0_0_1 + m_0_1 * ( src_pos_1_1_1 - src_pos_0_0_1 );
            VF src_pos_1_2_0 = src_pos_1_1_0 + m_1_2 * ( src_pos_2_2_0 - src_pos_1_1_0 );
            VF src_pos_1_2_1 = src_pos_1_1_1 + m_1_2 * ( src_pos_2_2_1 - src_pos_1_1_1 );

            VI ids = VI::gather( src_id_ptr, inds );

            VF::store( dst_0_pos_0_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_0 );
            VF::store( dst_0_pos_0_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_1 );
            VF::store( dst_0_pos_1_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_1_0 );
            VF::store( dst_0_pos_1_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_1_1 );
            VF::store( dst_0_pos_2_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_2_0 );
            VF::store( dst_0_pos_2_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_2_1 );
            VF::store( dst_0_pos_3_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_2_2_0 );
            VF::store( dst_0_pos_3_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_2_2_1 );

            VI::store( dst_0_id_ptr + 1 * beg_num_ind + 0 * simd_size.value, ids );
        } );

        nc_0.size += indices_size * 1;
    }
    template<class ShapeCoords>
    static void cut_l0_0_0_0_1_1_2_2_2_l0_0_0_2_2_2_3_0_3( const TI *indices_data, TI indices_size, ShapeCoords &nc_0, std::array<TI,4> dst_0_indices, const ShapeCoords &sc, std::array<TI,4> src_indices ) {
        Pos pos;
        Id id;

        const TF *src_pos_0_0_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 0 ].data;
        const TF *src_pos_0_0_1_ptr = sc[ pos ][ src_indices[ 0 ] ][ 1 ].data;
        const TF *src_pos_1_1_0_ptr = sc[ pos ][ src_indices[ 1 ] ][ 0 ].data;
        const TF *src_pos_1_1_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 1 ].data;
        const TF *src_pos_2_2_0_ptr = sc[ pos ][ src_indices[ 2 ] ][ 0 ].data;
        const TF *src_pos_2_2_1_ptr = sc[ pos ][ src_indices[ 2 ] ][ 1 ].data;
        const TF *src_pos_3_3_0_ptr = sc[ pos ][ src_indices[ 3 ] ][ 0 ].data;
        const TF *src_pos_3_3_1_ptr = sc[ pos ][ src_indices[ 3 ] ][ 1 ].data;

        const TI *src_id_ptr = sc[ id ].data;

        const TF *src_sp_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 2 ].data;
        const TF *src_sp_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 2 ].data;
        const TF *src_sp_2_ptr = sc[ pos ][ src_indices[ 2 ] ][ 2 ].data;
        const TF *src_sp_3_ptr = sc[ pos ][ src_indices[ 3 ] ][ 2 ].data;

        TF *dst_0_pos_0_0_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_0_1_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_1_0_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_1_1_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_2_0_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_2_1_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_3_0_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_3_1_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 1 ].data + nc_0.size;

        TI *dst_0_id_ptr = nc_0[ id ].data + nc_0.size;

        SimdRange<SimdSize<TF,Arch>::value>::for_each( indices_size, [&]( TI beg_num_ind, auto simd_size ) {
            using VI = SimdVec<TI,simd_size.value,Arch>;
            using VF = SimdVec<TF,simd_size.value,Arch>;

            VI inds = VI::load_aligned( indices_data + beg_num_ind );

            VF src_sp_0 = VF::gather( src_sp_0_ptr, inds );
            VF src_sp_1 = VF::gather( src_sp_1_ptr, inds );
            VF src_sp_2 = VF::gather( src_sp_2_ptr, inds );
            VF src_sp_3 = VF::gather( src_sp_3_ptr, inds );

            VF src_pos_0_0_0 = VF::gather( src_pos_0_0_0_ptr, inds );
            VF src_pos_0_0_1 = VF::gather( src_pos_0_0_1_ptr, inds );
            VF src_pos_1_1_0 = VF::gather( src_pos_1_1_0_ptr, inds );
            VF src_pos_1_1_1 = VF::gather( src_pos_1_1_1_ptr, inds );
            VF src_pos_2_2_0 = VF::gather( src_pos_2_2_0_ptr, inds );
            VF src_pos_2_2_1 = VF::gather( src_pos_2_2_1_ptr, inds );
            VF src_pos_3_3_0 = VF::gather( src_pos_3_3_0_ptr, inds );
            VF src_pos_3_3_1 = VF::gather( src_pos_3_3_1_ptr, inds );

            VF m_0_1 = src_sp_0 / ( src_sp_0 - src_sp_1 );
            VF m_0_3 = src_sp_0 / ( src_sp_0 - src_sp_3 );
            VF m_1_2 = src_sp_1 / ( src_sp_1 - src_sp_2 );
            VF m_2_3 = src_sp_2 / ( src_sp_2 - src_sp_3 );

            VF src_pos_0_1_0 = src_pos_0_0_0 + m_0_1 * ( src_pos_1_1_0 - src_pos_0_0_0 );
            VF src_pos_0_1_1 = src_pos_0_0_1 + m_0_1 * ( src_pos_1_1_1 - src_pos_0_0_1 );
            VF src_pos_0_3_0 = src_pos_0_0_0 + m_0_3 * ( src_pos_3_3_0 - src_pos_0_0_0 );
            VF src_pos_0_3_1 = src_pos_0_0_1 + m_0_3 * ( src_pos_3_3_1 - src_pos_0_0_1 );
            VF src_pos_1_2_0 = src_pos_1_1_0 + m_1_2 * ( src_pos_2_2_0 - src_pos_1_1_0 );
            VF src_pos_1_2_1 = src_pos_1_1_1 + m_1_2 * ( src_pos_2_2_1 - src_pos_1_1_1 );
            VF src_pos_2_3_0 = src_pos_2_2_0 + m_2_3 * ( src_pos_3_3_0 - src_pos_2_2_0 );
            VF src_pos_2_3_1 = src_pos_2_2_1 + m_2_3 * ( src_pos_3_3_1 - src_pos_2_2_1 );

            VI ids = VI::gather( src_id_ptr, inds );

            VF::store( dst_0_pos_0_0_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_0 );
            VF::store( dst_0_pos_0_1_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_1 );
            VF::store( dst_0_pos_1_0_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_0_1_0 );
            VF::store( dst_0_pos_1_1_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_0_1_1 );
            VF::store( dst_0_pos_2_0_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_1_2_0 );
            VF::store( dst_0_pos_2_1_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_1_2_1 );
            VF::store( dst_0_pos_3_0_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_2_2_0 );
            VF::store( dst_0_pos_3_1_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_2_2_1 );
            VF::store( dst_0_pos_0_0_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_0_0_0 );
            VF::store( dst_0_pos_0_1_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_0_0_1 );
            VF::store( dst_0_pos_1_0_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_2_2_0 );
            VF::store( dst_0_pos_1_1_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_2_2_1 );
            VF::store( dst_0_pos_2_0_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_2_3_0 );
            VF::store( dst_0_pos_2_1_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_2_3_1 );
            VF::store( dst_0_pos_3_0_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_0_3_0 );
            VF::store( dst_0_pos_3_1_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_0_3_1 );

            VI::store( dst_0_id_ptr + 2 * beg_num_ind + 0 * simd_size.value, ids );
            VI::store( dst_0_id_ptr + 2 * beg_num_ind + 1 * simd_size.value, ids );
        } );

        nc_0.size += indices_size * 2;
    }
    template<class ShapeCoords>
    static void cut_l0_0_0_0_1_1_2_2_2_l1_0_0_2_2_3_3( const TI *indices_data, TI indices_size, ShapeCoords &nc_0, std::array<TI,4> dst_0_indices, ShapeCoords &nc_1, std::array<TI,3> dst_1_indices, const ShapeCoords &sc, std::array<TI,4> src_indices ) {
        Pos pos;
        Id id;

        const TF *src_pos_0_0_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 0 ].data;
        const TF *src_pos_0_0_1_ptr = sc[ pos ][ src_indices[ 0 ] ][ 1 ].data;
        const TF *src_pos_1_1_0_ptr = sc[ pos ][ src_indices[ 1 ] ][ 0 ].data;
        const TF *src_pos_1_1_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 1 ].data;
        const TF *src_pos_2_2_0_ptr = sc[ pos ][ src_indices[ 2 ] ][ 0 ].data;
        const TF *src_pos_2_2_1_ptr = sc[ pos ][ src_indices[ 2 ] ][ 1 ].data;
        const TF *src_pos_3_3_0_ptr = sc[ pos ][ src_indices[ 3 ] ][ 0 ].data;
        const TF *src_pos_3_3_1_ptr = sc[ pos ][ src_indices[ 3 ] ][ 1 ].data;

        const TI *src_id_ptr = sc[ id ].data;

        const TF *src_sp_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 2 ].data;
        const TF *src_sp_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 2 ].data;
        const TF *src_sp_2_ptr = sc[ pos ][ src_indices[ 2 ] ][ 2 ].data;

        TF *dst_0_pos_0_0_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_0_1_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_1_0_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_1_1_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_2_0_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_2_1_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_3_0_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_3_1_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 1 ].data + nc_0.size;
        TF *dst_1_pos_0_0_ptr = nc_1[ pos ][ dst_1_indices[ 0 ] ][ 0 ].data + nc_1.size;
        TF *dst_1_pos_0_1_ptr = nc_1[ pos ][ dst_1_indices[ 0 ] ][ 1 ].data + nc_1.size;
        TF *dst_1_pos_1_0_ptr = nc_1[ pos ][ dst_1_indices[ 1 ] ][ 0 ].data + nc_1.size;
        TF *dst_1_pos_1_1_ptr = nc_1[ pos ][ dst_1_indices[ 1 ] ][ 1 ].data + nc_1.size;
        TF *dst_1_pos_2_0_ptr = nc_1[ pos ][ dst_1_indices[ 2 ] ][ 0 ].data + nc_1.size;
        TF *dst_1_pos_2_1_ptr = nc_1[ pos ][ dst_1_indices[ 2 ] ][ 1 ].data + nc_1.size;

        TI *dst_0_id_ptr = nc_0[ id ].data + nc_0.size;
        TI *dst_1_id_ptr = nc_1[ id ].data + nc_1.size;

        SimdRange<SimdSize<TF,Arch>::value>::for_each( indices_size, [&]( TI beg_num_ind, auto simd_size ) {
            using VI = SimdVec<TI,simd_size.value,Arch>;
            using VF = SimdVec<TF,simd_size.value,Arch>;

            VI inds = VI::load_aligned( indices_data + beg_num_ind );

            VF src_sp_0 = VF::gather( src_sp_0_ptr, inds );
            VF src_sp_1 = VF::gather( src_sp_1_ptr, inds );
            VF src_sp_2 = VF::gather( src_sp_2_ptr, inds );

            VF src_pos_0_0_0 = VF::gather( src_pos_0_0_0_ptr, inds );
            VF src_pos_0_0_1 = VF::gather( src_pos_0_0_1_ptr, inds );
            VF src_pos_1_1_0 = VF::gather( src_pos_1_1_0_ptr, inds );
            VF src_pos_1_1_1 = VF::gather( src_pos_1_1_1_ptr, inds );
            VF src_pos_2_2_0 = VF::gather( src_pos_2_2_0_ptr, inds );
            VF src_pos_2_2_1 = VF::gather( src_pos_2_2_1_ptr, inds );
            VF src_pos_3_3_0 = VF::gather( src_pos_3_3_0_ptr, inds );
            VF src_pos_3_3_1 = VF::gather( src_pos_3_3_1_ptr, inds );

            VF m_0_1 = src_sp_0 / ( src_sp_0 - src_sp_1 );
            VF m_1_2 = src_sp_1 / ( src_sp_1 - src_sp_2 );

            VF src_pos_0_1_0 = src_pos_0_0_0 + m_0_1 * ( src_pos_1_1_0 - src_pos_0_0_0 );
            VF src_pos_0_1_1 = src_pos_0_0_1 + m_0_1 * ( src_pos_1_1_1 - src_pos_0_0_1 );
            VF src_pos_1_2_0 = src_pos_1_1_0 + m_1_2 * ( src_pos_2_2_0 - src_pos_1_1_0 );
            VF src_pos_1_2_1 = src_pos_1_1_1 + m_1_2 * ( src_pos_2_2_1 - src_pos_1_1_1 );

            VI ids = VI::gather( src_id_ptr, inds );

            VF::store( dst_0_pos_0_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_0 );
            VF::store( dst_0_pos_0_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_1 );
            VF::store( dst_0_pos_1_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_1_0 );
            VF::store( dst_0_pos_1_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_1_1 );
            VF::store( dst_0_pos_2_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_2_0 );
            VF::store( dst_0_pos_2_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_2_1 );
            VF::store( dst_0_pos_3_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_2_2_0 );
            VF::store( dst_0_pos_3_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_2_2_1 );
            VF::store( dst_1_pos_0_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_0 );
            VF::store( dst_1_pos_0_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_1 );
            VF::store( dst_1_pos_1_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_2_2_0 );
            VF::store( dst_1_pos_1_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_2_2_1 );
            VF::store( dst_1_pos_2_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_3_3_0 );
            VF::store( dst_1_pos_2_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_3_3_1 );

            VI::store( dst_0_id_ptr + 1 * beg_num_ind + 0 * simd_size.value, ids );
            VI::store( dst_1_id_ptr + 1 * beg_num_ind + 0 * simd_size.value, ids );
        } );

        nc_0.size += indices_size * 1;
        nc_1.size += indices_size * 1;
    }
    template<class ShapeCoords>
    static void cut_l0_0_0_0_1_1_2_2_2_l1_3_3_0_0_2_2( const TI *indices_data, TI indices_size, ShapeCoords &nc_0, std::array<TI,4> dst_0_indices, ShapeCoords &nc_1, std::array<TI,3> dst_1_indices, const ShapeCoords &sc, std::array<TI,4> src_indices ) {
        Pos pos;
        Id id;

        const TF *src_pos_0_0_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 0 ].data;
        const TF *src_pos_0_0_1_ptr = sc[ pos ][ src_indices[ 0 ] ][ 1 ].data;
        const TF *src_pos_1_1_0_ptr = sc[ pos ][ src_indices[ 1 ] ][ 0 ].data;
        const TF *src_pos_1_1_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 1 ].data;
        const TF *src_pos_2_2_0_ptr = sc[ pos ][ src_indices[ 2 ] ][ 0 ].data;
        const TF *src_pos_2_2_1_ptr = sc[ pos ][ src_indices[ 2 ] ][ 1 ].data;
        const TF *src_pos_3_3_0_ptr = sc[ pos ][ src_indices[ 3 ] ][ 0 ].data;
        const TF *src_pos_3_3_1_ptr = sc[ pos ][ src_indices[ 3 ] ][ 1 ].data;

        const TI *src_id_ptr = sc[ id ].data;

        const TF *src_sp_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 2 ].data;
        const TF *src_sp_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 2 ].data;
        const TF *src_sp_2_ptr = sc[ pos ][ src_indices[ 2 ] ][ 2 ].data;

        TF *dst_0_pos_0_0_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_0_1_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_1_0_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_1_1_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_2_0_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_2_1_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_3_0_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_3_1_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 1 ].data + nc_0.size;
        TF *dst_1_pos_0_0_ptr = nc_1[ pos ][ dst_1_indices[ 0 ] ][ 0 ].data + nc_1.size;
        TF *dst_1_pos_0_1_ptr = nc_1[ pos ][ dst_1_indices[ 0 ] ][ 1 ].data + nc_1.size;
        TF *dst_1_pos_1_0_ptr = nc_1[ pos ][ dst_1_indices[ 1 ] ][ 0 ].data + nc_1.size;
        TF *dst_1_pos_1_1_ptr = nc_1[ pos ][ dst_1_indices[ 1 ] ][ 1 ].data + nc_1.size;
        TF *dst_1_pos_2_0_ptr = nc_1[ pos ][ dst_1_indices[ 2 ] ][ 0 ].data + nc_1.size;
        TF *dst_1_pos_2_1_ptr = nc_1[ pos ][ dst_1_indices[ 2 ] ][ 1 ].data + nc_1.size;

        TI *dst_0_id_ptr = nc_0[ id ].data + nc_0.size;
        TI *dst_1_id_ptr = nc_1[ id ].data + nc_1.size;

        SimdRange<SimdSize<TF,Arch>::value>::for_each( indices_size, [&]( TI beg_num_ind, auto simd_size ) {
            using VI = SimdVec<TI,simd_size.value,Arch>;
            using VF = SimdVec<TF,simd_size.value,Arch>;

            VI inds = VI::load_aligned( indices_data + beg_num_ind );

            VF src_sp_0 = VF::gather( src_sp_0_ptr, inds );
            VF src_sp_1 = VF::gather( src_sp_1_ptr, inds );
            VF src_sp_2 = VF::gather( src_sp_2_ptr, inds );

            VF src_pos_0_0_0 = VF::gather( src_pos_0_0_0_ptr, inds );
            VF src_pos_0_0_1 = VF::gather( src_pos_0_0_1_ptr, inds );
            VF src_pos_1_1_0 = VF::gather( src_pos_1_1_0_ptr, inds );
            VF src_pos_1_1_1 = VF::gather( src_pos_1_1_1_ptr, inds );
            VF src_pos_2_2_0 = VF::gather( src_pos_2_2_0_ptr, inds );
            VF src_pos_2_2_1 = VF::gather( src_pos_2_2_1_ptr, inds );
            VF src_pos_3_3_0 = VF::gather( src_pos_3_3_0_ptr, inds );
            VF src_pos_3_3_1 = VF::gather( src_pos_3_3_1_ptr, inds );

            VF m_0_1 = src_sp_0 / ( src_sp_0 - src_sp_1 );
            VF m_1_2 = src_sp_1 / ( src_sp_1 - src_sp_2 );

            VF src_pos_0_1_0 = src_pos_0_0_0 + m_0_1 * ( src_pos_1_1_0 - src_pos_0_0_0 );
            VF src_pos_0_1_1 = src_pos_0_0_1 + m_0_1 * ( src_pos_1_1_1 - src_pos_0_0_1 );
            VF src_pos_1_2_0 = src_pos_1_1_0 + m_1_2 * ( src_pos_2_2_0 - src_pos_1_1_0 );
            VF src_pos_1_2_1 = src_pos_1_1_1 + m_1_2 * ( src_pos_2_2_1 - src_pos_1_1_1 );

            VI ids = VI::gather( src_id_ptr, inds );

            VF::store( dst_0_pos_0_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_0 );
            VF::store( dst_0_pos_0_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_1 );
            VF::store( dst_0_pos_1_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_1_0 );
            VF::store( dst_0_pos_1_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_1_1 );
            VF::store( dst_0_pos_2_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_2_0 );
            VF::store( dst_0_pos_2_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_2_1 );
            VF::store( dst_0_pos_3_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_2_2_0 );
            VF::store( dst_0_pos_3_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_2_2_1 );
            VF::store( dst_1_pos_0_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_3_3_0 );
            VF::store( dst_1_pos_0_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_3_3_1 );
            VF::store( dst_1_pos_1_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_0 );
            VF::store( dst_1_pos_1_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_1 );
            VF::store( dst_1_pos_2_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_2_2_0 );
            VF::store( dst_1_pos_2_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_2_2_1 );

            VI::store( dst_0_id_ptr + 1 * beg_num_ind + 0 * simd_size.value, ids );
            VI::store( dst_1_id_ptr + 1 * beg_num_ind + 0 * simd_size.value, ids );
        } );

        nc_0.size += indices_size * 1;
        nc_1.size += indices_size * 1;
    }
    template<class ShapeCoords>
    static void cut_l0_0_0_0_1_2_3_2_2( const TI *indices_data, TI indices_size, ShapeCoords &nc_0, std::array<TI,4> dst_0_indices, const ShapeCoords &sc, std::array<TI,4> src_indices ) {
        Pos pos;
        Id id;

        const TF *src_pos_0_0_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 0 ].data;
        const TF *src_pos_0_0_1_ptr = sc[ pos ][ src_indices[ 0 ] ][ 1 ].data;
        const TF *src_pos_1_1_0_ptr = sc[ pos ][ src_indices[ 1 ] ][ 0 ].data;
        const TF *src_pos_1_1_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 1 ].data;
        const TF *src_pos_2_2_0_ptr = sc[ pos ][ src_indices[ 2 ] ][ 0 ].data;
        const TF *src_pos_2_2_1_ptr = sc[ pos ][ src_indices[ 2 ] ][ 1 ].data;
        const TF *src_pos_3_3_0_ptr = sc[ pos ][ src_indices[ 3 ] ][ 0 ].data;
        const TF *src_pos_3_3_1_ptr = sc[ pos ][ src_indices[ 3 ] ][ 1 ].data;

        const TI *src_id_ptr = sc[ id ].data;

        const TF *src_sp_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 2 ].data;
        const TF *src_sp_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 2 ].data;
        const TF *src_sp_2_ptr = sc[ pos ][ src_indices[ 2 ] ][ 2 ].data;
        const TF *src_sp_3_ptr = sc[ pos ][ src_indices[ 3 ] ][ 2 ].data;

        TF *dst_0_pos_0_0_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_0_1_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_1_0_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_1_1_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_2_0_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_2_1_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_3_0_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_3_1_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 1 ].data + nc_0.size;

        TI *dst_0_id_ptr = nc_0[ id ].data + nc_0.size;

        SimdRange<SimdSize<TF,Arch>::value>::for_each( indices_size, [&]( TI beg_num_ind, auto simd_size ) {
            using VI = SimdVec<TI,simd_size.value,Arch>;
            using VF = SimdVec<TF,simd_size.value,Arch>;

            VI inds = VI::load_aligned( indices_data + beg_num_ind );

            VF src_sp_0 = VF::gather( src_sp_0_ptr, inds );
            VF src_sp_1 = VF::gather( src_sp_1_ptr, inds );
            VF src_sp_2 = VF::gather( src_sp_2_ptr, inds );
            VF src_sp_3 = VF::gather( src_sp_3_ptr, inds );

            VF src_pos_0_0_0 = VF::gather( src_pos_0_0_0_ptr, inds );
            VF src_pos_0_0_1 = VF::gather( src_pos_0_0_1_ptr, inds );
            VF src_pos_1_1_0 = VF::gather( src_pos_1_1_0_ptr, inds );
            VF src_pos_1_1_1 = VF::gather( src_pos_1_1_1_ptr, inds );
            VF src_pos_2_2_0 = VF::gather( src_pos_2_2_0_ptr, inds );
            VF src_pos_2_2_1 = VF::gather( src_pos_2_2_1_ptr, inds );
            VF src_pos_3_3_0 = VF::gather( src_pos_3_3_0_ptr, inds );
            VF src_pos_3_3_1 = VF::gather( src_pos_3_3_1_ptr, inds );

            VF m_0_1 = src_sp_0 / ( src_sp_0 - src_sp_1 );
            VF m_2_3 = src_sp_2 / ( src_sp_2 - src_sp_3 );

            VF src_pos_0_1_0 = src_pos_0_0_0 + m_0_1 * ( src_pos_1_1_0 - src_pos_0_0_0 );
            VF src_pos_0_1_1 = src_pos_0_0_1 + m_0_1 * ( src_pos_1_1_1 - src_pos_0_0_1 );
            VF src_pos_2_3_0 = src_pos_2_2_0 + m_2_3 * ( src_pos_3_3_0 - src_pos_2_2_0 );
            VF src_pos_2_3_1 = src_pos_2_2_1 + m_2_3 * ( src_pos_3_3_1 - src_pos_2_2_1 );

            VI ids = VI::gather( src_id_ptr, inds );

            VF::store( dst_0_pos_0_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_0 );
            VF::store( dst_0_pos_0_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_1 );
            VF::store( dst_0_pos_1_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_1_0 );
            VF::store( dst_0_pos_1_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_1_1 );
            VF::store( dst_0_pos_2_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_2_3_0 );
            VF::store( dst_0_pos_2_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_2_3_1 );
            VF::store( dst_0_pos_3_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_2_2_0 );
            VF::store( dst_0_pos_3_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_2_2_1 );

            VI::store( dst_0_id_ptr + 1 * beg_num_ind + 0 * simd_size.value, ids );
        } );

        nc_0.size += indices_size * 1;
    }
    template<class ShapeCoords>
    static void cut_l0_0_0_1_1_0_2_1_2_l0_0_0_0_3_1_1_1_3( const TI *indices_data, TI indices_size, ShapeCoords &nc_0, std::array<TI,4> dst_0_indices, const ShapeCoords &sc, std::array<TI,4> src_indices ) {
        Pos pos;
        Id id;

        const TF *src_pos_0_0_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 0 ].data;
        const TF *src_pos_0_0_1_ptr = sc[ pos ][ src_indices[ 0 ] ][ 1 ].data;
        const TF *src_pos_1_1_0_ptr = sc[ pos ][ src_indices[ 1 ] ][ 0 ].data;
        const TF *src_pos_1_1_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 1 ].data;
        const TF *src_pos_2_2_0_ptr = sc[ pos ][ src_indices[ 2 ] ][ 0 ].data;
        const TF *src_pos_2_2_1_ptr = sc[ pos ][ src_indices[ 2 ] ][ 1 ].data;
        const TF *src_pos_3_3_0_ptr = sc[ pos ][ src_indices[ 3 ] ][ 0 ].data;
        const TF *src_pos_3_3_1_ptr = sc[ pos ][ src_indices[ 3 ] ][ 1 ].data;

        const TI *src_id_ptr = sc[ id ].data;

        const TF *src_sp_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 2 ].data;
        const TF *src_sp_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 2 ].data;
        const TF *src_sp_2_ptr = sc[ pos ][ src_indices[ 2 ] ][ 2 ].data;
        const TF *src_sp_3_ptr = sc[ pos ][ src_indices[ 3 ] ][ 2 ].data;

        TF *dst_0_pos_0_0_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_0_1_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_1_0_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_1_1_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_2_0_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_2_1_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_3_0_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_3_1_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 1 ].data + nc_0.size;

        TI *dst_0_id_ptr = nc_0[ id ].data + nc_0.size;

        SimdRange<SimdSize<TF,Arch>::value>::for_each( indices_size, [&]( TI beg_num_ind, auto simd_size ) {
            using VI = SimdVec<TI,simd_size.value,Arch>;
            using VF = SimdVec<TF,simd_size.value,Arch>;

            VI inds = VI::load_aligned( indices_data + beg_num_ind );

            VF src_sp_0 = VF::gather( src_sp_0_ptr, inds );
            VF src_sp_1 = VF::gather( src_sp_1_ptr, inds );
            VF src_sp_2 = VF::gather( src_sp_2_ptr, inds );
            VF src_sp_3 = VF::gather( src_sp_3_ptr, inds );

            VF src_pos_0_0_0 = VF::gather( src_pos_0_0_0_ptr, inds );
            VF src_pos_0_0_1 = VF::gather( src_pos_0_0_1_ptr, inds );
            VF src_pos_1_1_0 = VF::gather( src_pos_1_1_0_ptr, inds );
            VF src_pos_1_1_1 = VF::gather( src_pos_1_1_1_ptr, inds );
            VF src_pos_2_2_0 = VF::gather( src_pos_2_2_0_ptr, inds );
            VF src_pos_2_2_1 = VF::gather( src_pos_2_2_1_ptr, inds );
            VF src_pos_3_3_0 = VF::gather( src_pos_3_3_0_ptr, inds );
            VF src_pos_3_3_1 = VF::gather( src_pos_3_3_1_ptr, inds );

            VF m_0_2 = src_sp_0 / ( src_sp_0 - src_sp_2 );
            VF m_0_3 = src_sp_0 / ( src_sp_0 - src_sp_3 );
            VF m_1_2 = src_sp_1 / ( src_sp_1 - src_sp_2 );
            VF m_1_3 = src_sp_1 / ( src_sp_1 - src_sp_3 );

            VF src_pos_0_2_0 = src_pos_0_0_0 + m_0_2 * ( src_pos_2_2_0 - src_pos_0_0_0 );
            VF src_pos_0_2_1 = src_pos_0_0_1 + m_0_2 * ( src_pos_2_2_1 - src_pos_0_0_1 );
            VF src_pos_0_3_0 = src_pos_0_0_0 + m_0_3 * ( src_pos_3_3_0 - src_pos_0_0_0 );
            VF src_pos_0_3_1 = src_pos_0_0_1 + m_0_3 * ( src_pos_3_3_1 - src_pos_0_0_1 );
            VF src_pos_1_2_0 = src_pos_1_1_0 + m_1_2 * ( src_pos_2_2_0 - src_pos_1_1_0 );
            VF src_pos_1_2_1 = src_pos_1_1_1 + m_1_2 * ( src_pos_2_2_1 - src_pos_1_1_1 );
            VF src_pos_1_3_0 = src_pos_1_1_0 + m_1_3 * ( src_pos_3_3_0 - src_pos_1_1_0 );
            VF src_pos_1_3_1 = src_pos_1_1_1 + m_1_3 * ( src_pos_3_3_1 - src_pos_1_1_1 );

            VI ids = VI::gather( src_id_ptr, inds );

            VF::store( dst_0_pos_0_0_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_0 );
            VF::store( dst_0_pos_0_1_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_1 );
            VF::store( dst_0_pos_1_0_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_1_1_0 );
            VF::store( dst_0_pos_1_1_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_1_1_1 );
            VF::store( dst_0_pos_2_0_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_0_2_0 );
            VF::store( dst_0_pos_2_1_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_0_2_1 );
            VF::store( dst_0_pos_3_0_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_1_2_0 );
            VF::store( dst_0_pos_3_1_ptr + 2 * beg_num_ind + 0 * simd_size.value, src_pos_1_2_1 );
            VF::store( dst_0_pos_0_0_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_0_0_0 );
            VF::store( dst_0_pos_0_1_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_0_0_1 );
            VF::store( dst_0_pos_1_0_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_0_3_0 );
            VF::store( dst_0_pos_1_1_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_0_3_1 );
            VF::store( dst_0_pos_2_0_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_1_1_0 );
            VF::store( dst_0_pos_2_1_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_1_1_1 );
            VF::store( dst_0_pos_3_0_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_1_3_0 );
            VF::store( dst_0_pos_3_1_ptr + 2 * beg_num_ind + 1 * simd_size.value, src_pos_1_3_1 );

            VI::store( dst_0_id_ptr + 2 * beg_num_ind + 0 * simd_size.value, ids );
            VI::store( dst_0_id_ptr + 2 * beg_num_ind + 1 * simd_size.value, ids );
        } );

        nc_0.size += indices_size * 2;
    }
    template<class ShapeCoords>
    static void cut_l0_0_0_1_1_1_2_0_2( const TI *indices_data, TI indices_size, ShapeCoords &nc_0, std::array<TI,4> dst_0_indices, const ShapeCoords &sc, std::array<TI,3> src_indices ) {
        Pos pos;
        Id id;

        const TF *src_pos_0_0_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 0 ].data;
        const TF *src_pos_0_0_1_ptr = sc[ pos ][ src_indices[ 0 ] ][ 1 ].data;
        const TF *src_pos_1_1_0_ptr = sc[ pos ][ src_indices[ 1 ] ][ 0 ].data;
        const TF *src_pos_1_1_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 1 ].data;
        const TF *src_pos_2_2_0_ptr = sc[ pos ][ src_indices[ 2 ] ][ 0 ].data;
        const TF *src_pos_2_2_1_ptr = sc[ pos ][ src_indices[ 2 ] ][ 1 ].data;

        const TI *src_id_ptr = sc[ id ].data;

        const TF *src_sp_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 2 ].data;
        const TF *src_sp_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 2 ].data;
        const TF *src_sp_2_ptr = sc[ pos ][ src_indices[ 2 ] ][ 2 ].data;

        TF *dst_0_pos_0_0_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_0_1_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_1_0_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_1_1_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_2_0_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_2_1_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_3_0_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_3_1_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 1 ].data + nc_0.size;

        TI *dst_0_id_ptr = nc_0[ id ].data + nc_0.size;

        SimdRange<SimdSize<TF,Arch>::value>::for_each( indices_size, [&]( TI beg_num_ind, auto simd_size ) {
            using VI = SimdVec<TI,simd_size.value,Arch>;
            using VF = SimdVec<TF,simd_size.value,Arch>;

            VI inds = VI::load_aligned( indices_data + beg_num_ind );

            VF src_sp_0 = VF::gather( src_sp_0_ptr, inds );
            VF src_sp_1 = VF::gather( src_sp_1_ptr, inds );
            VF src_sp_2 = VF::gather( src_sp_2_ptr, inds );

            VF src_pos_0_0_0 = VF::gather( src_pos_0_0_0_ptr, inds );
            VF src_pos_0_0_1 = VF::gather( src_pos_0_0_1_ptr, inds );
            VF src_pos_1_1_0 = VF::gather( src_pos_1_1_0_ptr, inds );
            VF src_pos_1_1_1 = VF::gather( src_pos_1_1_1_ptr, inds );
            VF src_pos_2_2_0 = VF::gather( src_pos_2_2_0_ptr, inds );
            VF src_pos_2_2_1 = VF::gather( src_pos_2_2_1_ptr, inds );

            VF m_0_2 = src_sp_0 / ( src_sp_0 - src_sp_2 );
            VF m_1_2 = src_sp_1 / ( src_sp_1 - src_sp_2 );

            VF src_pos_0_2_0 = src_pos_0_0_0 + m_0_2 * ( src_pos_2_2_0 - src_pos_0_0_0 );
            VF src_pos_0_2_1 = src_pos_0_0_1 + m_0_2 * ( src_pos_2_2_1 - src_pos_0_0_1 );
            VF src_pos_1_2_0 = src_pos_1_1_0 + m_1_2 * ( src_pos_2_2_0 - src_pos_1_1_0 );
            VF src_pos_1_2_1 = src_pos_1_1_1 + m_1_2 * ( src_pos_2_2_1 - src_pos_1_1_1 );

            VI ids = VI::gather( src_id_ptr, inds );

            VF::store( dst_0_pos_0_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_0 );
            VF::store( dst_0_pos_0_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_1 );
            VF::store( dst_0_pos_1_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_1_0 );
            VF::store( dst_0_pos_1_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_1_1 );
            VF::store( dst_0_pos_2_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_2_0 );
            VF::store( dst_0_pos_2_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_2_1 );
            VF::store( dst_0_pos_3_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_2_0 );
            VF::store( dst_0_pos_3_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_2_1 );

            VI::store( dst_0_id_ptr + 1 * beg_num_ind + 0 * simd_size.value, ids );
        } );

        nc_0.size += indices_size * 1;
    }
    template<class ShapeCoords>
    static void cut_l0_0_0_1_1_1_2_0_2_l1_0_0_3_3_1_1( const TI *indices_data, TI indices_size, ShapeCoords &nc_0, std::array<TI,4> dst_0_indices, ShapeCoords &nc_1, std::array<TI,3> dst_1_indices, const ShapeCoords &sc, std::array<TI,4> src_indices ) {
        Pos pos;
        Id id;

        const TF *src_pos_0_0_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 0 ].data;
        const TF *src_pos_0_0_1_ptr = sc[ pos ][ src_indices[ 0 ] ][ 1 ].data;
        const TF *src_pos_1_1_0_ptr = sc[ pos ][ src_indices[ 1 ] ][ 0 ].data;
        const TF *src_pos_1_1_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 1 ].data;
        const TF *src_pos_2_2_0_ptr = sc[ pos ][ src_indices[ 2 ] ][ 0 ].data;
        const TF *src_pos_2_2_1_ptr = sc[ pos ][ src_indices[ 2 ] ][ 1 ].data;
        const TF *src_pos_3_3_0_ptr = sc[ pos ][ src_indices[ 3 ] ][ 0 ].data;
        const TF *src_pos_3_3_1_ptr = sc[ pos ][ src_indices[ 3 ] ][ 1 ].data;

        const TI *src_id_ptr = sc[ id ].data;

        const TF *src_sp_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 2 ].data;
        const TF *src_sp_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 2 ].data;
        const TF *src_sp_2_ptr = sc[ pos ][ src_indices[ 2 ] ][ 2 ].data;

        TF *dst_0_pos_0_0_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_0_1_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_1_0_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_1_1_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_2_0_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_2_1_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_3_0_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_3_1_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 1 ].data + nc_0.size;
        TF *dst_1_pos_0_0_ptr = nc_1[ pos ][ dst_1_indices[ 0 ] ][ 0 ].data + nc_1.size;
        TF *dst_1_pos_0_1_ptr = nc_1[ pos ][ dst_1_indices[ 0 ] ][ 1 ].data + nc_1.size;
        TF *dst_1_pos_1_0_ptr = nc_1[ pos ][ dst_1_indices[ 1 ] ][ 0 ].data + nc_1.size;
        TF *dst_1_pos_1_1_ptr = nc_1[ pos ][ dst_1_indices[ 1 ] ][ 1 ].data + nc_1.size;
        TF *dst_1_pos_2_0_ptr = nc_1[ pos ][ dst_1_indices[ 2 ] ][ 0 ].data + nc_1.size;
        TF *dst_1_pos_2_1_ptr = nc_1[ pos ][ dst_1_indices[ 2 ] ][ 1 ].data + nc_1.size;

        TI *dst_0_id_ptr = nc_0[ id ].data + nc_0.size;
        TI *dst_1_id_ptr = nc_1[ id ].data + nc_1.size;

        SimdRange<SimdSize<TF,Arch>::value>::for_each( indices_size, [&]( TI beg_num_ind, auto simd_size ) {
            using VI = SimdVec<TI,simd_size.value,Arch>;
            using VF = SimdVec<TF,simd_size.value,Arch>;

            VI inds = VI::load_aligned( indices_data + beg_num_ind );

            VF src_sp_0 = VF::gather( src_sp_0_ptr, inds );
            VF src_sp_1 = VF::gather( src_sp_1_ptr, inds );
            VF src_sp_2 = VF::gather( src_sp_2_ptr, inds );

            VF src_pos_0_0_0 = VF::gather( src_pos_0_0_0_ptr, inds );
            VF src_pos_0_0_1 = VF::gather( src_pos_0_0_1_ptr, inds );
            VF src_pos_1_1_0 = VF::gather( src_pos_1_1_0_ptr, inds );
            VF src_pos_1_1_1 = VF::gather( src_pos_1_1_1_ptr, inds );
            VF src_pos_2_2_0 = VF::gather( src_pos_2_2_0_ptr, inds );
            VF src_pos_2_2_1 = VF::gather( src_pos_2_2_1_ptr, inds );
            VF src_pos_3_3_0 = VF::gather( src_pos_3_3_0_ptr, inds );
            VF src_pos_3_3_1 = VF::gather( src_pos_3_3_1_ptr, inds );

            VF m_0_2 = src_sp_0 / ( src_sp_0 - src_sp_2 );
            VF m_1_2 = src_sp_1 / ( src_sp_1 - src_sp_2 );

            VF src_pos_0_2_0 = src_pos_0_0_0 + m_0_2 * ( src_pos_2_2_0 - src_pos_0_0_0 );
            VF src_pos_0_2_1 = src_pos_0_0_1 + m_0_2 * ( src_pos_2_2_1 - src_pos_0_0_1 );
            VF src_pos_1_2_0 = src_pos_1_1_0 + m_1_2 * ( src_pos_2_2_0 - src_pos_1_1_0 );
            VF src_pos_1_2_1 = src_pos_1_1_1 + m_1_2 * ( src_pos_2_2_1 - src_pos_1_1_1 );

            VI ids = VI::gather( src_id_ptr, inds );

            VF::store( dst_0_pos_0_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_0 );
            VF::store( dst_0_pos_0_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_1 );
            VF::store( dst_0_pos_1_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_1_0 );
            VF::store( dst_0_pos_1_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_1_1 );
            VF::store( dst_0_pos_2_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_2_0 );
            VF::store( dst_0_pos_2_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_2_1 );
            VF::store( dst_0_pos_3_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_2_0 );
            VF::store( dst_0_pos_3_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_2_1 );
            VF::store( dst_1_pos_0_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_0 );
            VF::store( dst_1_pos_0_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_0_0_1 );
            VF::store( dst_1_pos_1_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_3_3_0 );
            VF::store( dst_1_pos_1_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_3_3_1 );
            VF::store( dst_1_pos_2_0_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_1_0 );
            VF::store( dst_1_pos_2_1_ptr + 1 * beg_num_ind + 0 * simd_size.value, src_pos_1_1_1 );

            VI::store( dst_0_id_ptr + 1 * beg_num_ind + 0 * simd_size.value, ids );
            VI::store( dst_1_id_ptr + 1 * beg_num_ind + 0 * simd_size.value, ids );
        } );

        nc_0.size += indices_size * 1;
        nc_1.size += indices_size * 1;
    }
    template<class ShapeCoords>
    static void cut_l0_0_0_1_1_1_2_0_3( const TI *indices_data, TI indices_size, ShapeCoords &nc_0, std::array<TI,4> dst_0_indices, const ShapeCoords &sc, std::array<TI,4> src_indices ) {
        Pos pos;
        Id id;

        const TF *src_pos_0_0_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 0 ].data;
        const TF *src_pos_0_0_1_ptr = sc[ pos ][ src_indices[ 0 ] ][ 1 ].data;
        const TF *src_pos_1_1_0_ptr = sc[ pos ][ src_indices[ 1 ] ][ 0 ].data;
        const TF *src_pos_1_1_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 1 ].data;
        const TF *src_pos_2_2_0_ptr = sc[ pos ][ src_indices[ 2 ] ][ 0 ].data;
        const TF *src_pos_2_2_1_ptr = sc[ pos ][ src_indices[ 2 ] ][ 1 ].data;
        const TF *src_pos_3_3_0_ptr = sc[ pos ][ src_indices[ 3 ] ][ 0 ].data;
        const TF *src_pos_3_3_1_ptr = sc[ pos ][ src_indices[ 3 ] ][ 1 ].data;

        const TI *src_id_ptr = sc[ id ].data;

        const TF *src_sp_0_ptr = sc[ pos ][ src_indices[ 0 ] ][ 2 ].data;
        const TF *src_sp_1_ptr = sc[ pos ][ src_indices[ 1 ] ][ 2 ].data;
        const TF *src_sp_2_ptr = sc[ pos ][ src_indices[ 2 ] ][ 2 ].data;
        const TF *src_sp_3_ptr = sc[ pos ][ src_indices[ 3 ] ][ 2 ].data;

        TF *dst_0_pos_0_0_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_0_1_ptr = nc_0[ pos ][ dst_0_indices[ 0 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_1_0_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_1_1_ptr = nc_0[ pos ][ dst_0_indices[ 1 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_2_0_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_2_1_ptr = nc_0[ pos ][ dst_0_indices[ 2 ] ][ 1 ].data + nc_0.size;
        TF *dst_0_pos_3_0_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 0 ].data + nc_0.size;
        TF *dst_0_pos_3_1_ptr = nc_0[ pos ][ dst_0_indices[ 3 ] ][ 1 ].data + nc_0.size;

        TI *dst_0_id_ptr = nc_0[ id ].data + nc_0.size;

        SimdRange<SimdSize<TF,Arch>::value>::for_each( indices_size, [&]( TI beg_num_ind, auto simd_size ) {
            using VI = SimdVec<TI,simd_size.value,Arch>;
            using VF = SimdVec<TF,simd_size.value,Arch>;

            VI inds = VI::load_aligned( indices_data + beg_num_ind );

            VF src_sp_0 = VF::gather( src_sp_0_ptr, inds );
            VF src_sp_1 = VF::gather( src_sp_1_ptr, inds );
            VF src_sp_2 = VF::gather( src_sp_2_ptr, inds );
            VF src_sp_3 = VF::gather( src_sp_3_ptr, inds );

            VF src_pos_0_0_0 = VF::gather( src_pos_0_0_0_ptr, inds );
            VF src_pos_0_0_1 = VF::gather( src_pos_0_0_1_ptr, inds );
            VF src_pos_1_1_0 = VF::gather( src_pos_1_1_0_ptr, inds );
            VF src_pos_1_1_1 = VF::gather( src_pos_1_1_1_ptr, inds );
            VF src_pos_2_2_0 = VF::gather( src_pos_2_2_0_ptr, inds );
            VF src_pos_2_2_1 = VF::gather( src_pos_2_2_1_ptr, inds );
            VF src_pos_3_3_0 = VF::gather( src_pos_3_3_0_ptr, inds );
            VF src_pos_3_3_1 = VF::gather( src_pos_3_3_1_ptr, inds );

            VF m_0_3 = src_sp_0 / ( src_sp_0 - src_sp_3 );
            VF m_1_2 = src_sp_1 / ( src_sp_1 - src_sp_