if ( dim == 2 && name == "3" ) {
  for( TI be = 0; be < sc.size; be += cut_chunk_size ) {
    make_sp_and_cases( dirs, sps, sc, be, N<3>(), { { "3", { 1, 2, 2, 1, 2, 1, 1, 0 } } } );

    using RVO = RecursivePolyhedronCutVecOp_2<TF,TI,Arch,Pos,Id>;
    RVO::cut_l0_0_0_1_1_2_2( tmp_indices_bcc.data() + 0 * cut_chunk_size, tmp_offsets_bcc[ 0 ] - 0 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 1, 2 } );
    RVO::cut_l0_0_0_0_1_1_2_l0_0_0_1_2_2_2( tmp_indices_bcc.data() + 1 * cut_chunk_size, tmp_offsets_bcc[ 1 ] - 1 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 2, 1 }, sc, { 1, 0, 2 } );
    RVO::cut_l0_0_0_0_1_1_2_l0_0_0_1_2_2_2( tmp_indices_bcc.data() + 2 * cut_chunk_size, tmp_offsets_bcc[ 2 ] - 2 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 1, 2 } );
    RVO::cut_l0_0_0_0_1_0_2( tmp_indices_bcc.data() + 3 * cut_chunk_size, tmp_offsets_bcc[ 3 ] - 3 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 2, 0, 1 } );
    RVO::cut_l0_0_0_0_1_1_2_l0_0_0_1_2_2_2( tmp_indices_bcc.data() + 4 * cut_chunk_size, tmp_offsets_bcc[ 4 ] - 4 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 2, 1 }, sc, { 0, 2, 1 } );
    RVO::cut_l0_0_0_0_1_0_2( tmp_indices_bcc.data() + 5 * cut_chunk_size, tmp_offsets_bcc[ 5 ] - 5 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 1, 2, 0 } );
    RVO::cut_l0_0_0_0_1_0_2( tmp_indices_bcc.data() + 6 * cut_chunk_size, tmp_offsets_bcc[ 6 ] - 6 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 1, 2 } );
  };
  continue;
}
