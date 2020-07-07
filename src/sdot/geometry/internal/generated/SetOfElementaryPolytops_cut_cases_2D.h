if ( dim == 2 && name == "3" ) {
    for( TI be = 0; be < sc.size; be += cut_chunk_size ) {
        make_sp_and_cases( dirs, sps, sc, be, N<3>(), { { "3", { 1, 0, 0, 1, 0, 1, 1, 0 } }, { "4", { 0, 1, 1, 0, 1, 0, 0, 0 } } } );

        //    using RVO = RecursivePolyhedronCutVecOp_2<TF,TI,Arch,Pos,Id>;
        //    RVO::cut_l0_0_0_1_1_2_2( beg_cut_cases[ 0 ], nb_cut_cases[ 0 ], shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 1, 2 } );
        //    RVO::cut_l0_0_0_1_1_1_2_0_2( beg_cut_cases[ 1 ], nb_cut_cases[ 1 ], shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 1, 2, 0 } );
        //    RVO::cut_l0_0_0_0_1_1_2_2_2( beg_cut_cases[ 2 ], nb_cut_cases[ 2 ], shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 0, 1, 2 } );
        //    RVO::cut_l0_0_0_0_1_0_2( beg_cut_cases[ 3 ], nb_cut_cases[ 3 ], shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 2, 0, 1 } );
        //    RVO::cut_l0_0_0_1_1_1_2_0_2( beg_cut_cases[ 4 ], nb_cut_cases[ 4 ], shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 0, 1, 2 } );
        //    RVO::cut_l0_0_0_0_1_0_2( beg_cut_cases[ 5 ], nb_cut_cases[ 5 ], shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 1, 2, 0 } );
        //    RVO::cut_l0_0_0_0_1_0_2( beg_cut_cases[ 6 ], nb_cut_cases[ 6 ], shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 1, 2 } );
    }
    continue;
}
if ( dim == 2 && name == "4" ) {
    // make_sp_and_cases( dirs, sps, sc, N<4>(), { { "3", { 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0 } }, { "4", { 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 2, 0, 1, 0, 0, 0 } } } );

    //    using RVO = RecursivePolyhedronCutVecOp_2<TF,TI,Arch,Pos,Id>;
    //    RVO::cut_l0_0_0_1_1_2_2_3_3( beg_cut_cases[ 0 ], nb_cut_cases[ 0 ], shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 0, 1, 2, 3 } );
    //    RVO::cut_l0_0_0_1_1_1_2_0_2_l1_0_0_3_3_1_1( beg_cut_cases[ 1 ], nb_cut_cases[ 1 ], shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 1, 3, 0, 2 } );
    //    RVO::cut_l0_0_0_0_1_1_2_2_2_l1_0_0_2_2_3_3( beg_cut_cases[ 2 ], nb_cut_cases[ 2 ], shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 1, 2, 3 } );
    //    RVO::cut_l0_0_0_1_1_1_2_0_3( beg_cut_cases[ 3 ], nb_cut_cases[ 3 ], shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 2, 3, 0, 1 } );
    //    RVO::cut_l0_0_0_0_1_1_2_2_2_l1_3_3_0_0_2_2( beg_cut_cases[ 4 ], nb_cut_cases[ 4 ], shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 1, 2, 3, 0 } );
    //    RVO::cut_l0_0_0_0_1_1_2_2_2_l0_0_0_2_2_2_3_0_3( beg_cut_cases[ 5 ], nb_cut_cases[ 5 ], shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 1, 2, 3, 0 } );
    //    RVO::cut_l0_0_0_0_1_2_3_2_2( beg_cut_cases[ 6 ], nb_cut_cases[ 6 ], shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 0, 1, 3, 2 } );
    //    RVO::cut_l0_0_0_0_1_0_2( beg_cut_cases[ 7 ], nb_cut_cases[ 7 ], shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 3, 0, 2 } );
    //    RVO::cut_l0_0_0_1_1_1_2_0_2_l1_0_0_3_3_1_1( beg_cut_cases[ 8 ], nb_cut_cases[ 8 ], shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 2, 3, 1 } );
    //    RVO::cut_l0_0_0_1_1_1_2_0_3( beg_cut_cases[ 9 ], nb_cut_cases[ 9 ], shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 1, 2, 3, 0 } );
    //    RVO::cut_l0_0_0_1_1_0_2_1_2_l0_0_0_0_3_1_1_1_3( beg_cut_cases[ 10 ], nb_cut_cases[ 10 ], shape_list( tmp_shape_map, "4" ), { 0, 3, 1, 2 }, sc, { 0, 2, 1, 3 } );
    //    RVO::cut_l0_0_0_0_1_0_2( beg_cut_cases[ 11 ], nb_cut_cases[ 11 ], shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 2, 3, 1 } );
    //    RVO::cut_l0_0_0_1_1_1_2_0_3( beg_cut_cases[ 12 ], nb_cut_cases[ 12 ], shape_list( tmp_shape_map, "4" ), { 0, 1, 2, 3 }, sc, { 0, 1, 2, 3 } );
    //    RVO::cut_l0_0_0_0_1_0_2( beg_cut_cases[ 13 ], nb_cut_cases[ 13 ], shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 1, 2, 0 } );
    //    RVO::cut_l0_0_0_0_1_0_2( beg_cut_cases[ 14 ], nb_cut_cases[ 14 ], shape_list( tmp_shape_map, "3" ), { 0, 1, 2 }, sc, { 0, 1, 3 } );
    continue;
}
