void mk_items_n3_0_0_1_1_2_2_f3_0_1_2( ShapeData &nsd_0, const std::array<BI,3> &nni_0, const std::array<BI,3> &nfi_0, const ShapeData &osd, const std::array<BI,3> &oni, const std::array<BI,3> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];

        new_x_0_0[ nsd_0.size ] = x_0_0;
        new_y_0_0[ nsd_0.size ] = y_0_0;
        new_x_1_0[ nsd_0.size ] = x_1_1;
        new_y_1_0[ nsd_0.size ] = y_1_1;
        new_x_2_0[ nsd_0.size ] = x_2_2;
        new_y_2_0[ nsd_0.size ] = y_2_2;

        new_f_0_0[ nsd_0.size ] = old_f_0[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_2_0[ nsd_0.size ] = old_f_2[ off ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
void mk_items_n4_0_0_1_1_2_2_3_3_f4_0_1_2_3( ShapeData &nsd_0, const std::array<BI,4> &nni_0, const std::array<BI,4> &nfi_0, const ShapeData &osd, const std::array<BI,4> &oni, const std::array<BI,4> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;
    TI *new_f_3_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 3 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;
    const TF *old_x_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 0 ) * osd.rese;
    const TF *old_y_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;
    const TI *old_f_3 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 3 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_3_3 = old_x_3[ off ];
        TF y_3_3 = old_y_3[ off ];

        new_x_0_0[ nsd_0.size ] = x_0_0;
        new_y_0_0[ nsd_0.size ] = y_0_0;
        new_x_1_0[ nsd_0.size ] = x_1_1;
        new_y_1_0[ nsd_0.size ] = y_1_1;
        new_x_2_0[ nsd_0.size ] = x_2_2;
        new_y_2_0[ nsd_0.size ] = y_2_2;
        new_x_3_0[ nsd_0.size ] = x_3_3;
        new_y_3_0[ nsd_0.size ] = y_3_3;

        new_f_0_0[ nsd_0.size ] = old_f_0[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_2_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_3_0[ nsd_0.size ] = old_f_3[ off ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
void mk_items_n3_0_1_1_1_1_2_f3_0_1_c( ShapeData &nsd_0, const std::array<BI,3> &nni_0, const std::array<BI,3> &nfi_0, const ShapeData &osd, const std::array<BI,3> &oni, const std::array<BI,2> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_0_1 = x_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( x_1_1 - x_0_0 );
        TF y_0_1 = y_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( y_1_1 - y_0_0 );
        TF x_1_2 = x_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( x_2_2 - x_1_1 );
        TF y_1_2 = y_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( y_2_2 - y_1_1 );

        new_x_0_0[ nsd_0.size ] = x_0_1;
        new_y_0_0[ nsd_0.size ] = y_0_1;
        new_x_1_0[ nsd_0.size ] = x_1_1;
        new_y_1_0[ nsd_0.size ] = y_1_1;
        new_x_2_0[ nsd_0.size ] = x_1_2;
        new_y_2_0[ nsd_0.size ] = y_1_2;

        new_f_0_0[ nsd_0.size ] = old_f_0[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_2_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
void mk_items_n4_0_1_1_1_2_2_2_0_f4_0_1_2_c( ShapeData &nsd_0, const std::array<BI,4> &nni_0, const std::array<BI,4> &nfi_0, const ShapeData &osd, const std::array<BI,3> &oni, const std::array<BI,3> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;
    TI *new_f_3_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 3 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_0_1 = x_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( x_1_1 - x_0_0 );
        TF y_0_1 = y_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( y_1_1 - y_0_0 );
        TF x_2_0 = x_2_2 + scp_2 / ( scp_2 - scp_0 ) * ( x_0_0 - x_2_2 );
        TF y_2_0 = y_2_2 + scp_2 / ( scp_2 - scp_0 ) * ( y_0_0 - y_2_2 );

        new_x_0_0[ nsd_0.size ] = x_0_1;
        new_y_0_0[ nsd_0.size ] = y_0_1;
        new_x_1_0[ nsd_0.size ] = x_1_1;
        new_y_1_0[ nsd_0.size ] = y_1_1;
        new_x_2_0[ nsd_0.size ] = x_2_2;
        new_y_2_0[ nsd_0.size ] = y_2_2;
        new_x_3_0[ nsd_0.size ] = x_2_0;
        new_y_3_0[ nsd_0.size ] = y_2_0;

        new_f_0_0[ nsd_0.size ] = old_f_0[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_2_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_3_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
void mk_items_n4_0_1_1_1_2_2_2_3_f4_0_1_2_c( ShapeData &nsd_0, const std::array<BI,4> &nni_0, const std::array<BI,4> &nfi_0, const ShapeData &osd, const std::array<BI,4> &oni, const std::array<BI,3> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;
    TI *new_f_3_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 3 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;
    const TF *old_x_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 0 ) * osd.rese;
    const TF *old_y_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;
    const TF *old_scp_3 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 3 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];
    TF scp_3 = old_scp_3[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_3_3 = old_x_3[ off ];
        TF y_3_3 = old_y_3[ off ];
        TF x_0_1 = x_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( x_1_1 - x_0_0 );
        TF y_0_1 = y_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( y_1_1 - y_0_0 );
        TF x_2_3 = x_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( x_3_3 - x_2_2 );
        TF y_2_3 = y_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( y_3_3 - y_2_2 );

        new_x_0_0[ nsd_0.size ] = x_0_1;
        new_y_0_0[ nsd_0.size ] = y_0_1;
        new_x_1_0[ nsd_0.size ] = x_1_1;
        new_y_1_0[ nsd_0.size ] = y_1_1;
        new_x_2_0[ nsd_0.size ] = x_2_2;
        new_y_2_0[ nsd_0.size ] = y_2_2;
        new_x_3_0[ nsd_0.size ] = x_2_3;
        new_y_3_0[ nsd_0.size ] = y_2_3;

        new_f_0_0[ nsd_0.size ] = old_f_0[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_2_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_3_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
void mk_items_n4_1_2_2_2_0_0_0_1_f4_1_2_0_c( ShapeData &nsd_0, const std::array<BI,4> &nni_0, const std::array<BI,4> &nfi_0, const ShapeData &osd, const std::array<BI,3> &oni, const std::array<BI,3> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;
    TI *new_f_3_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 3 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_0_1 = x_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( x_1_1 - x_0_0 );
        TF y_0_1 = y_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( y_1_1 - y_0_0 );
        TF x_1_2 = x_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( x_2_2 - x_1_1 );
        TF y_1_2 = y_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( y_2_2 - y_1_1 );

        new_x_0_0[ nsd_0.size ] = x_1_2;
        new_y_0_0[ nsd_0.size ] = y_1_2;
        new_x_1_0[ nsd_0.size ] = x_2_2;
        new_y_1_0[ nsd_0.size ] = y_2_2;
        new_x_2_0[ nsd_0.size ] = x_0_0;
        new_y_2_0[ nsd_0.size ] = y_0_0;
        new_x_3_0[ nsd_0.size ] = x_0_1;
        new_y_3_0[ nsd_0.size ] = y_0_1;

        new_f_0_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_2_0[ nsd_0.size ] = old_f_0[ off ];
        new_f_3_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
void mk_items_n3_1_2_2_2_2_0_f3_1_2_c( ShapeData &nsd_0, const std::array<BI,3> &nni_0, const std::array<BI,3> &nfi_0, const ShapeData &osd, const std::array<BI,3> &oni, const std::array<BI,3> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_1_2 = x_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( x_2_2 - x_1_1 );
        TF y_1_2 = y_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( y_2_2 - y_1_1 );
        TF x_2_0 = x_2_2 + scp_2 / ( scp_2 - scp_0 ) * ( x_0_0 - x_2_2 );
        TF y_2_0 = y_2_2 + scp_2 / ( scp_2 - scp_0 ) * ( y_0_0 - y_2_2 );

        new_x_0_0[ nsd_0.size ] = x_1_2;
        new_y_0_0[ nsd_0.size ] = y_1_2;
        new_x_1_0[ nsd_0.size ] = x_2_2;
        new_y_1_0[ nsd_0.size ] = y_2_2;
        new_x_2_0[ nsd_0.size ] = x_2_0;
        new_y_2_0[ nsd_0.size ] = y_2_0;

        new_f_0_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_2_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
void mk_items_n3_1_2_2_2_2_3_f3_1_2_c( ShapeData &nsd_0, const std::array<BI,3> &nni_0, const std::array<BI,3> &nfi_0, const ShapeData &osd, const std::array<BI,4> &oni, const std::array<BI,3> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;
    const TF *old_x_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 0 ) * osd.rese;
    const TF *old_y_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;
    const TF *old_scp_3 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 3 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];
    TF scp_3 = old_scp_3[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_3_3 = old_x_3[ off ];
        TF y_3_3 = old_y_3[ off ];
        TF x_1_2 = x_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( x_2_2 - x_1_1 );
        TF y_1_2 = y_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( y_2_2 - y_1_1 );
        TF x_2_3 = x_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( x_3_3 - x_2_2 );
        TF y_2_3 = y_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( y_3_3 - y_2_2 );

        new_x_0_0[ nsd_0.size ] = x_1_2;
        new_y_0_0[ nsd_0.size ] = y_1_2;
        new_x_1_0[ nsd_0.size ] = x_2_2;
        new_y_1_0[ nsd_0.size ] = y_2_2;
        new_x_2_0[ nsd_0.size ] = x_2_3;
        new_y_2_0[ nsd_0.size ] = y_2_3;

        new_f_0_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_2_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
void mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n3_0_1_1_1_1_2_f3_0_1_c( ShapeData &nsd_0, const std::array<BI,3> &nni_0, const std::array<BI,3> &nfi_0, ShapeData &nsd_1, const std::array<BI,3> &nni_1, const std::array<BI,3> &nfi_1, const ShapeData &osd, const std::array<BI,4> &oni, const std::array<BI,3> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_0_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 0 ] * dim + 0 ) * nsd_1.rese;
    TF *new_y_0_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 0 ] * dim + 1 ) * nsd_1.rese;
    TF *new_x_1_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 1 ] * dim + 0 ) * nsd_1.rese;
    TF *new_y_1_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 1 ] * dim + 1 ) * nsd_1.rese;
    TF *new_x_2_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 2 ] * dim + 0 ) * nsd_1.rese;
    TF *new_y_2_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 2 ] * dim + 1 ) * nsd_1.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;
    TI *new_f_0_1 = reinterpret_cast<TI *>( nsd_1.face_ids ) + nni_1[ 0 ] * nsd_1.rese;
    TI *new_f_1_1 = reinterpret_cast<TI *>( nsd_1.face_ids ) + nni_1[ 1 ] * nsd_1.rese;
    TI *new_f_2_1 = reinterpret_cast<TI *>( nsd_1.face_ids ) + nni_1[ 2 ] * nsd_1.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );
    TI *new_ids_1 = reinterpret_cast<TI *>( nsd_1.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;
    const TF *old_x_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 0 ) * osd.rese;
    const TF *old_y_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;
    const TF *old_scp_3 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 3 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];
    TF scp_3 = old_scp_3[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_3_3 = old_x_3[ off ];
        TF y_3_3 = old_y_3[ off ];
        TF x_0_1 = x_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( x_1_1 - x_0_0 );
        TF y_0_1 = y_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( y_1_1 - y_0_0 );
        TF x_1_2 = x_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( x_2_2 - x_1_1 );
        TF y_1_2 = y_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( y_2_2 - y_1_1 );
        TF x_2_3 = x_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( x_3_3 - x_2_2 );
        TF y_2_3 = y_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( y_3_3 - y_2_2 );

        new_x_0_0[ nsd_0.size ] = x_1_2;
        new_y_0_0[ nsd_0.size ] = y_1_2;
        new_x_1_0[ nsd_0.size ] = x_2_2;
        new_y_1_0[ nsd_0.size ] = y_2_2;
        new_x_2_0[ nsd_0.size ] = x_2_3;
        new_y_2_0[ nsd_0.size ] = y_2_3;
        new_x_0_1[ nsd_1.size ] = x_0_1;
        new_y_0_1[ nsd_1.size ] = y_0_1;
        new_x_1_1[ nsd_1.size ] = x_1_1;
        new_y_1_1[ nsd_1.size ] = y_1_1;
        new_x_2_1[ nsd_1.size ] = x_1_2;
        new_y_2_1[ nsd_1.size ] = y_1_2;

        new_f_0_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_2_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];
        new_f_0_1[ nsd_1.size ] = old_f_0[ off ];
        new_f_1_1[ nsd_1.size ] = old_f_1[ off ];
        new_f_2_1[ nsd_1.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];
        new_ids_1[ nsd_1.size ] = old_ids[ off ];

        ++nsd_0.size;
        ++nsd_1.size;
    }
}
void mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n3_1_2_2_2_2_3_f3_1_2_c( ShapeData &nsd_0, const std::array<BI,3> &nni_0, const std::array<BI,3> &nfi_0, ShapeData &nsd_1, const std::array<BI,3> &nni_1, const std::array<BI,3> &nfi_1, const ShapeData &osd, const std::array<BI,4> &oni, const std::array<BI,3> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_0_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 0 ] * dim + 0 ) * nsd_1.rese;
    TF *new_y_0_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 0 ] * dim + 1 ) * nsd_1.rese;
    TF *new_x_1_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 1 ] * dim + 0 ) * nsd_1.rese;
    TF *new_y_1_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 1 ] * dim + 1 ) * nsd_1.rese;
    TF *new_x_2_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 2 ] * dim + 0 ) * nsd_1.rese;
    TF *new_y_2_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 2 ] * dim + 1 ) * nsd_1.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;
    TI *new_f_0_1 = reinterpret_cast<TI *>( nsd_1.face_ids ) + nni_1[ 0 ] * nsd_1.rese;
    TI *new_f_1_1 = reinterpret_cast<TI *>( nsd_1.face_ids ) + nni_1[ 1 ] * nsd_1.rese;
    TI *new_f_2_1 = reinterpret_cast<TI *>( nsd_1.face_ids ) + nni_1[ 2 ] * nsd_1.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );
    TI *new_ids_1 = reinterpret_cast<TI *>( nsd_1.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;
    const TF *old_x_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 0 ) * osd.rese;
    const TF *old_y_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;
    const TF *old_scp_3 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 3 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];
    TF scp_3 = old_scp_3[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_3_3 = old_x_3[ off ];
        TF y_3_3 = old_y_3[ off ];
        TF x_1_2 = x_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( x_2_2 - x_1_1 );
        TF y_1_2 = y_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( y_2_2 - y_1_1 );
        TF x_2_3 = x_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( x_3_3 - x_2_2 );
        TF y_2_3 = y_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( y_3_3 - y_2_2 );

        new_x_0_0[ nsd_0.size ] = x_1_2;
        new_y_0_0[ nsd_0.size ] = y_1_2;
        new_x_1_0[ nsd_0.size ] = x_2_2;
        new_y_1_0[ nsd_0.size ] = y_2_2;
        new_x_2_0[ nsd_0.size ] = x_2_3;
        new_y_2_0[ nsd_0.size ] = y_2_3;
        new_x_0_1[ nsd_1.size ] = x_1_2;
        new_y_0_1[ nsd_1.size ] = y_1_2;
        new_x_1_1[ nsd_1.size ] = x_2_2;
        new_y_1_1[ nsd_1.size ] = y_2_2;
        new_x_2_1[ nsd_1.size ] = x_2_3;
        new_y_2_1[ nsd_1.size ] = y_2_3;

        new_f_0_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_2_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];
        new_f_0_1[ nsd_1.size ] = old_f_1[ off ];
        new_f_1_1[ nsd_1.size ] = old_f_2[ off ];
        new_f_2_1[ nsd_1.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];
        new_ids_1[ nsd_1.size ] = old_ids[ off ];

        ++nsd_0.size;
        ++nsd_1.size;
    }
}
void mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n3_2_3_3_3_3_0_f3_2_3_c( ShapeData &nsd_0, const std::array<BI,3> &nni_0, const std::array<BI,3> &nfi_0, ShapeData &nsd_1, const std::array<BI,3> &nni_1, const std::array<BI,3> &nfi_1, const ShapeData &osd, const std::array<BI,4> &oni, const std::array<BI,4> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_0_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 0 ] * dim + 0 ) * nsd_1.rese;
    TF *new_y_0_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 0 ] * dim + 1 ) * nsd_1.rese;
    TF *new_x_1_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 1 ] * dim + 0 ) * nsd_1.rese;
    TF *new_y_1_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 1 ] * dim + 1 ) * nsd_1.rese;
    TF *new_x_2_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 2 ] * dim + 0 ) * nsd_1.rese;
    TF *new_y_2_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 2 ] * dim + 1 ) * nsd_1.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;
    TI *new_f_0_1 = reinterpret_cast<TI *>( nsd_1.face_ids ) + nni_1[ 0 ] * nsd_1.rese;
    TI *new_f_1_1 = reinterpret_cast<TI *>( nsd_1.face_ids ) + nni_1[ 1 ] * nsd_1.rese;
    TI *new_f_2_1 = reinterpret_cast<TI *>( nsd_1.face_ids ) + nni_1[ 2 ] * nsd_1.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );
    TI *new_ids_1 = reinterpret_cast<TI *>( nsd_1.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;
    const TF *old_x_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 0 ) * osd.rese;
    const TF *old_y_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;
    const TI *old_f_3 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 3 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;
    const TF *old_scp_3 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 3 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];
    TF scp_3 = old_scp_3[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_3_3 = old_x_3[ off ];
        TF y_3_3 = old_y_3[ off ];
        TF x_1_2 = x_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( x_2_2 - x_1_1 );
        TF y_1_2 = y_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( y_2_2 - y_1_1 );
        TF x_2_3 = x_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( x_3_3 - x_2_2 );
        TF y_2_3 = y_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( y_3_3 - y_2_2 );
        TF x_3_0 = x_3_3 + scp_3 / ( scp_3 - scp_0 ) * ( x_0_0 - x_3_3 );
        TF y_3_0 = y_3_3 + scp_3 / ( scp_3 - scp_0 ) * ( y_0_0 - y_3_3 );

        new_x_0_0[ nsd_0.size ] = x_1_2;
        new_y_0_0[ nsd_0.size ] = y_1_2;
        new_x_1_0[ nsd_0.size ] = x_2_2;
        new_y_1_0[ nsd_0.size ] = y_2_2;
        new_x_2_0[ nsd_0.size ] = x_2_3;
        new_y_2_0[ nsd_0.size ] = y_2_3;
        new_x_0_1[ nsd_1.size ] = x_2_3;
        new_y_0_1[ nsd_1.size ] = y_2_3;
        new_x_1_1[ nsd_1.size ] = x_3_3;
        new_y_1_1[ nsd_1.size ] = y_3_3;
        new_x_2_1[ nsd_1.size ] = x_3_0;
        new_y_2_1[ nsd_1.size ] = y_3_0;

        new_f_0_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_2_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];
        new_f_0_1[ nsd_1.size ] = old_f_2[ off ];
        new_f_1_1[ nsd_1.size ] = old_f_3[ off ];
        new_f_2_1[ nsd_1.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];
        new_ids_1[ nsd_1.size ] = old_ids[ off ];

        ++nsd_0.size;
        ++nsd_1.size;
    }
}
void mk_items_n3_1_2_2_2_2_3_f3_1_2_c_n3_3_0_0_0_0_1_f3_3_0_c( ShapeData &nsd_0, const std::array<BI,3> &nni_0, const std::array<BI,3> &nfi_0, ShapeData &nsd_1, const std::array<BI,3> &nni_1, const std::array<BI,3> &nfi_1, const ShapeData &osd, const std::array<BI,4> &oni, const std::array<BI,4> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_0_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 0 ] * dim + 0 ) * nsd_1.rese;
    TF *new_y_0_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 0 ] * dim + 1 ) * nsd_1.rese;
    TF *new_x_1_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 1 ] * dim + 0 ) * nsd_1.rese;
    TF *new_y_1_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 1 ] * dim + 1 ) * nsd_1.rese;
    TF *new_x_2_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 2 ] * dim + 0 ) * nsd_1.rese;
    TF *new_y_2_1 = reinterpret_cast<TF *>( nsd_1.coordinates ) + ( nni_1[ 2 ] * dim + 1 ) * nsd_1.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;
    TI *new_f_0_1 = reinterpret_cast<TI *>( nsd_1.face_ids ) + nni_1[ 0 ] * nsd_1.rese;
    TI *new_f_1_1 = reinterpret_cast<TI *>( nsd_1.face_ids ) + nni_1[ 1 ] * nsd_1.rese;
    TI *new_f_2_1 = reinterpret_cast<TI *>( nsd_1.face_ids ) + nni_1[ 2 ] * nsd_1.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );
    TI *new_ids_1 = reinterpret_cast<TI *>( nsd_1.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;
    const TF *old_x_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 0 ) * osd.rese;
    const TF *old_y_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;
    const TI *old_f_3 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 3 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;
    const TF *old_scp_3 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 3 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];
    TF scp_3 = old_scp_3[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_3_3 = old_x_3[ off ];
        TF y_3_3 = old_y_3[ off ];
        TF x_0_1 = x_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( x_1_1 - x_0_0 );
        TF y_0_1 = y_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( y_1_1 - y_0_0 );
        TF x_1_2 = x_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( x_2_2 - x_1_1 );
        TF y_1_2 = y_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( y_2_2 - y_1_1 );
        TF x_2_3 = x_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( x_3_3 - x_2_2 );
        TF y_2_3 = y_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( y_3_3 - y_2_2 );
        TF x_3_0 = x_3_3 + scp_3 / ( scp_3 - scp_0 ) * ( x_0_0 - x_3_3 );
        TF y_3_0 = y_3_3 + scp_3 / ( scp_3 - scp_0 ) * ( y_0_0 - y_3_3 );

        new_x_0_0[ nsd_0.size ] = x_1_2;
        new_y_0_0[ nsd_0.size ] = y_1_2;
        new_x_1_0[ nsd_0.size ] = x_2_2;
        new_y_1_0[ nsd_0.size ] = y_2_2;
        new_x_2_0[ nsd_0.size ] = x_2_3;
        new_y_2_0[ nsd_0.size ] = y_2_3;
        new_x_0_1[ nsd_1.size ] = x_3_0;
        new_y_0_1[ nsd_1.size ] = y_3_0;
        new_x_1_1[ nsd_1.size ] = x_0_0;
        new_y_1_1[ nsd_1.size ] = y_0_0;
        new_x_2_1[ nsd_1.size ] = x_0_1;
        new_y_2_1[ nsd_1.size ] = y_0_1;

        new_f_0_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_2_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];
        new_f_0_1[ nsd_1.size ] = old_f_3[ off ];
        new_f_1_1[ nsd_1.size ] = old_f_0[ off ];
        new_f_2_1[ nsd_1.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];
        new_ids_1[ nsd_1.size ] = old_ids[ off ];

        ++nsd_0.size;
        ++nsd_1.size;
    }
}
void mk_items_n4_1_2_2_2_3_3_3_0_f4_1_2_3_c( ShapeData &nsd_0, const std::array<BI,4> &nni_0, const std::array<BI,4> &nfi_0, const ShapeData &osd, const std::array<BI,4> &oni, const std::array<BI,4> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;
    TI *new_f_3_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 3 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;
    const TF *old_x_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 0 ) * osd.rese;
    const TF *old_y_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;
    const TI *old_f_3 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 3 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;
    const TF *old_scp_3 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 3 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];
    TF scp_3 = old_scp_3[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_3_3 = old_x_3[ off ];
        TF y_3_3 = old_y_3[ off ];
        TF x_1_2 = x_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( x_2_2 - x_1_1 );
        TF y_1_2 = y_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( y_2_2 - y_1_1 );
        TF x_3_0 = x_3_3 + scp_3 / ( scp_3 - scp_0 ) * ( x_0_0 - x_3_3 );
        TF y_3_0 = y_3_3 + scp_3 / ( scp_3 - scp_0 ) * ( y_0_0 - y_3_3 );

        new_x_0_0[ nsd_0.size ] = x_1_2;
        new_y_0_0[ nsd_0.size ] = y_1_2;
        new_x_1_0[ nsd_0.size ] = x_2_2;
        new_y_1_0[ nsd_0.size ] = y_2_2;
        new_x_2_0[ nsd_0.size ] = x_3_3;
        new_y_2_0[ nsd_0.size ] = y_3_3;
        new_x_3_0[ nsd_0.size ] = x_3_0;
        new_y_3_0[ nsd_0.size ] = y_3_0;

        new_f_0_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_2_0[ nsd_0.size ] = old_f_3[ off ];
        new_f_3_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
void mk_items_n3_2_0_0_0_0_1_f3_2_0_c( ShapeData &nsd_0, const std::array<BI,3> &nni_0, const std::array<BI,3> &nfi_0, const ShapeData &osd, const std::array<BI,3> &oni, const std::array<BI,3> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_0_1 = x_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( x_1_1 - x_0_0 );
        TF y_0_1 = y_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( y_1_1 - y_0_0 );
        TF x_2_0 = x_2_2 + scp_2 / ( scp_2 - scp_0 ) * ( x_0_0 - x_2_2 );
        TF y_2_0 = y_2_2 + scp_2 / ( scp_2 - scp_0 ) * ( y_0_0 - y_2_2 );

        new_x_0_0[ nsd_0.size ] = x_2_0;
        new_y_0_0[ nsd_0.size ] = y_2_0;
        new_x_1_0[ nsd_0.size ] = x_0_0;
        new_y_1_0[ nsd_0.size ] = y_0_0;
        new_x_2_0[ nsd_0.size ] = x_0_1;
        new_y_2_0[ nsd_0.size ] = y_0_1;

        new_f_0_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_0[ off ];
        new_f_2_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
void mk_items_n4_2_0_0_0_1_1_1_2_f4_2_0_1_c( ShapeData &nsd_0, const std::array<BI,4> &nni_0, const std::array<BI,4> &nfi_0, const ShapeData &osd, const std::array<BI,3> &oni, const std::array<BI,3> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;
    TI *new_f_3_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 3 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_1_2 = x_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( x_2_2 - x_1_1 );
        TF y_1_2 = y_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( y_2_2 - y_1_1 );
        TF x_2_0 = x_2_2 + scp_2 / ( scp_2 - scp_0 ) * ( x_0_0 - x_2_2 );
        TF y_2_0 = y_2_2 + scp_2 / ( scp_2 - scp_0 ) * ( y_0_0 - y_2_2 );

        new_x_0_0[ nsd_0.size ] = x_2_0;
        new_y_0_0[ nsd_0.size ] = y_2_0;
        new_x_1_0[ nsd_0.size ] = x_0_0;
        new_y_1_0[ nsd_0.size ] = y_0_0;
        new_x_2_0[ nsd_0.size ] = x_1_1;
        new_y_2_0[ nsd_0.size ] = y_1_1;
        new_x_3_0[ nsd_0.size ] = x_1_2;
        new_y_3_0[ nsd_0.size ] = y_1_2;

        new_f_0_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_0[ off ];
        new_f_2_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_3_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
void mk_items_n4_2_3_3_3_0_0_0_1_f4_2_3_0_c( ShapeData &nsd_0, const std::array<BI,4> &nni_0, const std::array<BI,4> &nfi_0, const ShapeData &osd, const std::array<BI,4> &oni, const std::array<BI,4> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;
    TI *new_f_3_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 3 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;
    const TF *old_x_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 0 ) * osd.rese;
    const TF *old_y_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;
    const TI *old_f_3 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 3 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;
    const TF *old_scp_3 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 3 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];
    TF scp_3 = old_scp_3[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_3_3 = old_x_3[ off ];
        TF y_3_3 = old_y_3[ off ];
        TF x_0_1 = x_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( x_1_1 - x_0_0 );
        TF y_0_1 = y_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( y_1_1 - y_0_0 );
        TF x_2_3 = x_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( x_3_3 - x_2_2 );
        TF y_2_3 = y_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( y_3_3 - y_2_2 );

        new_x_0_0[ nsd_0.size ] = x_2_3;
        new_y_0_0[ nsd_0.size ] = y_2_3;
        new_x_1_0[ nsd_0.size ] = x_3_3;
        new_y_1_0[ nsd_0.size ] = y_3_3;
        new_x_2_0[ nsd_0.size ] = x_0_0;
        new_y_2_0[ nsd_0.size ] = y_0_0;
        new_x_3_0[ nsd_0.size ] = x_0_1;
        new_y_3_0[ nsd_0.size ] = y_0_1;

        new_f_0_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_3[ off ];
        new_f_2_0[ nsd_0.size ] = old_f_0[ off ];
        new_f_3_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
void mk_items_n3_2_3_3_3_3_0_f3_2_3_c( ShapeData &nsd_0, const std::array<BI,3> &nni_0, const std::array<BI,3> &nfi_0, const ShapeData &osd, const std::array<BI,4> &oni, const std::array<BI,4> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;
    const TF *old_x_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 0 ) * osd.rese;
    const TF *old_y_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;
    const TI *old_f_3 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 3 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;
    const TF *old_scp_3 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 3 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_2 = old_scp_2[ off ];
    TF scp_3 = old_scp_3[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_3_3 = old_x_3[ off ];
        TF y_3_3 = old_y_3[ off ];
        TF x_2_3 = x_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( x_3_3 - x_2_2 );
        TF y_2_3 = y_2_2 + scp_2 / ( scp_2 - scp_3 ) * ( y_3_3 - y_2_2 );
        TF x_3_0 = x_3_3 + scp_3 / ( scp_3 - scp_0 ) * ( x_0_0 - x_3_3 );
        TF y_3_0 = y_3_3 + scp_3 / ( scp_3 - scp_0 ) * ( y_0_0 - y_3_3 );

        new_x_0_0[ nsd_0.size ] = x_2_3;
        new_y_0_0[ nsd_0.size ] = y_2_3;
        new_x_1_0[ nsd_0.size ] = x_3_3;
        new_y_1_0[ nsd_0.size ] = y_3_3;
        new_x_2_0[ nsd_0.size ] = x_3_0;
        new_y_2_0[ nsd_0.size ] = y_3_0;

        new_f_0_0[ nsd_0.size ] = old_f_2[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_3[ off ];
        new_f_2_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
void mk_items_n3_3_0_0_0_0_1_f3_3_0_c( ShapeData &nsd_0, const std::array<BI,3> &nni_0, const std::array<BI,3> &nfi_0, const ShapeData &osd, const std::array<BI,4> &oni, const std::array<BI,4> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;
    const TF *old_x_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 0 ) * osd.rese;
    const TF *old_y_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;
    const TI *old_f_3 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 3 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;
    const TF *old_scp_3 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 3 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_1 = old_scp_1[ off ];
    TF scp_3 = old_scp_3[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_3_3 = old_x_3[ off ];
        TF y_3_3 = old_y_3[ off ];
        TF x_0_1 = x_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( x_1_1 - x_0_0 );
        TF y_0_1 = y_0_0 + scp_0 / ( scp_0 - scp_1 ) * ( y_1_1 - y_0_0 );
        TF x_3_0 = x_3_3 + scp_3 / ( scp_3 - scp_0 ) * ( x_0_0 - x_3_3 );
        TF y_3_0 = y_3_3 + scp_3 / ( scp_3 - scp_0 ) * ( y_0_0 - y_3_3 );

        new_x_0_0[ nsd_0.size ] = x_3_0;
        new_y_0_0[ nsd_0.size ] = y_3_0;
        new_x_1_0[ nsd_0.size ] = x_0_0;
        new_y_1_0[ nsd_0.size ] = y_0_0;
        new_x_2_0[ nsd_0.size ] = x_0_1;
        new_y_2_0[ nsd_0.size ] = y_0_1;

        new_f_0_0[ nsd_0.size ] = old_f_3[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_0[ off ];
        new_f_2_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
void mk_items_n4_3_0_0_0_1_1_1_2_f4_3_0_1_c( ShapeData &nsd_0, const std::array<BI,4> &nni_0, const std::array<BI,4> &nfi_0, const ShapeData &osd, const std::array<BI,4> &oni, const std::array<BI,4> &ofi, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;
    TI *new_f_3_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 3 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;
    const TF *old_x_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 0 ) * osd.rese;
    const TF *old_y_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;
    const TI *old_f_3 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 3 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *indices = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::indices ] );

    const TF *old_scp_0 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 0 ] * osd.rese;
    const TF *old_scp_1 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 1 ] * osd.rese;
    const TF *old_scp_2 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 2 ] * osd.rese;
    const TF *old_scp_3 = reinterpret_cast<const TF *>( osd.tmp[ ShapeData::out_scps ] ) + oni[ 3 ] * osd.rese;

    for( BI num_ind = osd.case_offsets[ num_case + 0 ]; num_ind < osd.case_offsets[ num_case + 1 ]; ++num_ind ) {
        TI off = indices[ num_ind ];

    TF scp_0 = old_scp_0[ off ];
    TF scp_1 = old_scp_1[ off ];
    TF scp_2 = old_scp_2[ off ];
    TF scp_3 = old_scp_3[ off ];

        TF x_0_0 = old_x_0[ off ];
        TF y_0_0 = old_y_0[ off ];
        TF x_1_1 = old_x_1[ off ];
        TF y_1_1 = old_y_1[ off ];
        TF x_2_2 = old_x_2[ off ];
        TF y_2_2 = old_y_2[ off ];
        TF x_3_3 = old_x_3[ off ];
        TF y_3_3 = old_y_3[ off ];
        TF x_1_2 = x_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( x_2_2 - x_1_1 );
        TF y_1_2 = y_1_1 + scp_1 / ( scp_1 - scp_2 ) * ( y_2_2 - y_1_1 );
        TF x_3_0 = x_3_3 + scp_3 / ( scp_3 - scp_0 ) * ( x_0_0 - x_3_3 );
        TF y_3_0 = y_3_3 + scp_3 / ( scp_3 - scp_0 ) * ( y_0_0 - y_3_3 );

        new_x_0_0[ nsd_0.size ] = x_3_0;
        new_y_0_0[ nsd_0.size ] = y_3_0;
        new_x_1_0[ nsd_0.size ] = x_0_0;
        new_y_1_0[ nsd_0.size ] = y_0_0;
        new_x_2_0[ nsd_0.size ] = x_1_1;
        new_y_2_0[ nsd_0.size ] = y_1_1;
        new_x_3_0[ nsd_0.size ] = x_1_2;
        new_y_3_0[ nsd_0.size ] = y_1_2;

        new_f_0_0[ nsd_0.size ] = old_f_3[ off ];
        new_f_1_0[ nsd_0.size ] = old_f_0[ off ];
        new_f_2_0[ nsd_0.size ] = old_f_1[ off ];
        new_f_3_0[ nsd_0.size ] = reinterpret_cast<const TI *>( cut_ids )[ old_ids[ off ] ];

        new_ids_0[ nsd_0.size ] = old_ids[ off ];

        ++nsd_0.size;
    }
}
