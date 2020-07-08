if ( dim == 2 && name == "3" ) {
  for( TI be = 0; be < sc.size; be += cut_chunk_size ) {
    make_sp_and_cases( dirs, sps, sc, be, N<3>(), { { "3", { 1, 0, 0, 1, 0, 1, 1, 0 } }, { "4", { 0, 1, 1, 0, 1, 0, 0, 0 } } } );

    using RVO = RecursivePolyhedronCutVecOp_2<TF,TI,Arch,Pos,Id>;
    RVO::cut_l0_0_0_1_1_2_2( tmp_indices_bcc.data() + 0 * cut_chunk_size, tmp_offsets_bcc[ 0 ] - 0 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 1, 2 } );
    RVO::cut_l0_0_0_1_1_1_2_0_2( tmp_indices_bcc.data() + 1 * cut_chunk_size, tmp_offsets_bcc[ 1 ] - 1 * cut_chunk_size, shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 1, 2, 0 } );
    RVO::cut_l0_0_0_0_1_1_2_2_2( tmp_indices_bcc.data() + 2 * cut_chunk_size, tmp_offsets_bcc[ 2 ] - 2 * cut_chunk_size, shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 0, 1, 2 } );
    RVO::cut_l0_0_0_0_1_0_2( tmp_indices_bcc.data() + 3 * cut_chunk_size, tmp_offsets_bcc[ 3 ] - 3 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 2, 0, 1 } );
    RVO::cut_l0_0_0_1_1_1_2_0_2( tmp_indices_bcc.data() + 4 * cut_chunk_size, tmp_offsets_bcc[ 4 ] - 4 * cut_chunk_size, shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 0, 1, 2 } );
    RVO::cut_l0_0_0_0_1_0_2( tmp_indices_bcc.data() + 5 * cut_chunk_size, tmp_offsets_bcc[ 5 ] - 5 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 1, 2, 0 } );
    RVO::cut_l0_0_0_0_1_0_2( tmp_indices_bcc.data() + 6 * cut_chunk_size, tmp_offsets_bcc[ 6 ] - 6 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 1, 2 } );
  };
  continue;
}
if ( dim == 2 && name == "4" ) {
  for( TI be = 0; be < sc.size; be += cut_chunk_size ) {
    make_sp_and_cases( dirs, sps, sc, be, N<4>(), { { "3", { 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0 } }, { "4", { 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 2, 0, 1, 0, 0, 0 } } } );

    using RVO = RecursivePolyhedronCutVecOp_2<TF,TI,Arch,Pos,Id>;
    RVO::cut_l0_0_0_1_1_2_2_3_3( tmp_indices_bcc.data() + 0 * cut_chunk_size, tmp_offsets_bcc[ 0 ] - 0 * cut_chunk_size, shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 0, 1, 2, 3 } );
    RVO::cut_l0_0_0_1_1_1_2_0_2_l1_0_0_3_3_1_1( tmp_indices_bcc.data() + 1 * cut_chunk_size, tmp_offsets_bcc[ 1 ] - 1 * cut_chunk_size, shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 1, 3, 0, 2 } );
    RVO::cut_l0_0_0_0_1_1_2_2_2_l1_0_0_2_2_3_3( tmp_indices_bcc.data() + 2 * cut_chunk_size, tmp_offsets_bcc[ 2 ] - 2 * cut_chunk_size, shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 1, 2, 3 } );
    RVO::cut_l0_0_0_1_1_1_2_0_3( tmp_indices_bcc.data() + 3 * cut_chunk_size, tmp_offsets_bcc[ 3 ] - 3 * cut_chunk_size, shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 2, 3, 0, 1 } );
    RVO::cut_l0_0_0_0_1_1_2_2_2_l1_3_3_0_0_2_2( tmp_indices_bcc.data() + 4 * cut_chunk_size, tmp_offsets_bcc[ 4 ] - 4 * cut_chunk_size, shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 1, 2, 3, 0 } );
    RVO::cut_l0_0_0_0_1_1_2_2_2_l0_0_0_2_2_2_3_0_3( tmp_indices_bcc.data() + 5 * cut_chunk_size, tmp_offsets_bcc[ 5 ] - 5 * cut_chunk_size, shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 1, 2, 3, 0 } );
    RVO::cut_l0_0_0_0_1_2_3_2_2( tmp_indices_bcc.data() + 6 * cut_chunk_size, tmp_offsets_bcc[ 6 ] - 6 * cut_chunk_size, shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 0, 1, 3, 2 } );
    RVO::cut_l0_0_0_0_1_0_2( tmp_indices_bcc.data() + 7 * cut_chunk_size, tmp_offsets_bcc[ 7 ] - 7 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 3, 0, 2 } );
    RVO::cut_l0_0_0_1_1_1_2_0_2_l1_0_0_3_3_1_1( tmp_indices_bcc.data() + 8 * cut_chunk_size, tmp_offsets_bcc[ 8 ] - 8 * cut_chunk_size, shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 2, 3, 1 } );
    RVO::cut_l0_0_0_1_1_1_2_0_3( tmp_indices_bcc.data() + 9 * cut_chunk_size, tmp_offsets_bcc[ 9 ] - 9 * cut_chunk_size, shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 1, 2, 3, 0 } );
    RVO::cut_l0_0_0_1_1_0_2_1_2_l0_0_0_0_3_1_1_1_3( tmp_indices_bcc.data() + 10 * cut_chunk_size, tmp_offsets_bcc[ 10 ] - 10 * cut_chunk_size, shape_list( tmp_shape_map, "4" ), { 0, 3, 1, 2 }, sc, { 0, 2, 1, 3 } );
    RVO::cut_l0_0_0_0_1_0_2( tmp_indices_bcc.data() + 11 * cut_chunk_size, tmp_offsets_bcc[ 11 ] - 11 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 2, 3, 1 } );
    RVO::cut_l0_0_0_1_1_1_2_0_3( tmp_indices_bcc.data() + 12 * cut_chunk_size, tmp_offsets_bcc[ 12 ] - 12 * cut_chunk_size, shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 0, 1, 2, 3 } );
    RVO::cut_l0_0_0_0_1_0_2( tmp_indices_bcc.data() + 13 * cut_chunk_size, tmp_offsets_bcc[ 13 ] - 13 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 1, 2, 0 } );
    RVO::cut_l0_0_0_0_1_0_2( tmp_indices_bcc.data() + 14 * cut_chunk_size, tmp_offsets_bcc[ 14 ] - 14 * cut_chunk_size, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 1, 3 } );
  };
  continue;
}
