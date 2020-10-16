void mk_items( const ShapeData &osd, const std::array<BI,0> &oni, BI num_case, const void *cut_ids, N<2> dim ) override {





    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *o0 = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::offset_0 ] );
    const TI *o1 = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::offset_1 ] );
    const TI *cc = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::cut_case ] );

    for( BI nmp = 0; nmp < nb_multiprocs(); ++nmp ) {
        for( BI ind = o0[ num_case * nb_multiprocs() + nmp ], end = o1[ num_case * nb_multiprocs() + nmp ]; ind < end; ++ind ) {
            TI off = cc[ ind ];





        }
    }
}
void mk_items_0_0_1_1_2_2( ShapeData &nsd_0, const std::array<BI,3> &nni_0, const ShapeData &osd, const std::array<BI,3> &oni, BI num_case, const void *cut_ids, N<2> dim ) override {
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

    const TI *o0 = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::offset_0 ] );
    const TI *o1 = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::offset_1 ] );
    const TI *cc = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::cut_case ] );

    for( BI nmp = 0; nmp < nb_multiprocs(); ++nmp ) {
        for( BI ind = o0[ num_case * nb_multiprocs() + nmp ], end = o1[ num_case * nb_multiprocs() + nmp ]; ind < end; ++ind ) {
            TI off = cc[ ind ];

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
}
void mk_items_0_0_1_1_2_2_3_3( ShapeData &nsd_0, const std::array<BI,4> &nni_0, const ShapeData &osd, const std::array<BI,4> &oni, BI num_case, const void *cut_ids, N<2> dim ) override {
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

    const TI *o0 = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::offset_0 ] );
    const TI *o1 = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::offset_1 ] );
    const TI *cc = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::cut_case ] );

    for( BI nmp = 0; nmp < nb_multiprocs(); ++nmp ) {
        for( BI ind = o0[ num_case * nb_multiprocs() + nmp ], end = o1[ num_case * nb_multiprocs() + nmp ]; ind < end; ++ind ) {
            TI off = cc[ ind ];

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
}
void mk_items_0_0_1_1_2_2_3_3_4_4( ShapeData &nsd_0, const std::array<BI,5> &nni_0, const ShapeData &osd, const std::array<BI,5> &oni, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_0_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 0 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_1_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 1 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_2_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 2 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_3_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 3 ] * dim + 1 ) * nsd_0.rese;
    TF *new_x_4_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 4 ] * dim + 0 ) * nsd_0.rese;
    TF *new_y_4_0 = reinterpret_cast<TF *>( nsd_0.coordinates ) + ( nni_0[ 4 ] * dim + 1 ) * nsd_0.rese;

    TI *new_f_0_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 0 ] * nsd_0.rese;
    TI *new_f_1_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 1 ] * nsd_0.rese;
    TI *new_f_2_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 2 ] * nsd_0.rese;
    TI *new_f_3_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 3 ] * nsd_0.rese;
    TI *new_f_4_0 = reinterpret_cast<TI *>( nsd_0.face_ids ) + nni_0[ 4 ] * nsd_0.rese;

    TI *new_ids_0 = reinterpret_cast<TI *>( nsd_0.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 0 ) * osd.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 0 ] * dim + 1 ) * osd.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 0 ) * osd.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 1 ] * dim + 1 ) * osd.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 0 ) * osd.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 2 ] * dim + 1 ) * osd.rese;
    const TF *old_x_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 0 ) * osd.rese;
    const TF *old_y_3 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 3 ] * dim + 1 ) * osd.rese;
    const TF *old_x_4 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 4 ] * dim + 0 ) * osd.rese;
    const TF *old_y_4 = reinterpret_cast<const TF *>( osd.coordinates ) + ( oni[ 4 ] * dim + 1 ) * osd.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 0 ] * osd.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 1 ] * osd.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 2 ] * osd.rese;
    const TI *old_f_3 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 3 ] * osd.rese;
    const TI *old_f_4 = reinterpret_cast<const TI *>( osd.face_ids ) + oni[ 4 ] * osd.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( osd.ids );

    const TI *o0 = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::offset_0 ] );
    const TI *o1 = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::offset_1 ] );
    const TI *cc = reinterpret_cast<const TI *>( osd.tmp[ ShapeData::cut_case ] );

    for( BI nmp = 0; nmp < nb_multiprocs(); ++nmp ) {
        for( BI ind = o0[ num_case * nb_multiprocs() + nmp ], end = o1[ num_case * nb_multiprocs() + nmp ]; ind < end; ++ind ) {
            TI off = cc[ ind ];

            TF x_0_0 = old_x_0[ off ];
            TF y_0_0 = old_y_0[ off ];
            TF x_1_1 = old_x_1[ off ];
            TF y_1_1 = old_y_1[ off ];
            TF x_2_2 = old_x_2[ off ];
            TF y_2_2 = old_y_2[ off ];
            TF x_3_3 = old_x_3[ off ];
            TF y_3_3 = old_y_3[ off ];
            TF x_4_4 = old_x_4[ off ];
            TF y_4_4 = old_y_4[ off ];

            new_x_0_0[ nsd_0.size ] = x_0_0;
            new_y_0_0[ nsd_0.size ] = y_0_0;
            new_x_1_0[ nsd_0.size ] = x_1_1;
            new_y_1_0[ nsd_0.size ] = y_1_1;
            new_x_2_0[ nsd_0.size ] = x_2_2;
            new_y_2_0[ nsd_0.size ] = y_2_2;
            new_x_3_0[ nsd_0.size ] = x_3_3;
            new_y_3_0[ nsd_0.size ] = y_3_3;
            new_x_4_0[ nsd_0.size ] = x_4_4;
            new_y_4_0[ nsd_0.size ] = y_4_4;

            new_f_0_0[ nsd_0.size ] = old_f_0[ off ];
            new_f_1_0[ nsd_0.size ] = old_f_1[ off ];
            new_f_2_0[ nsd_0.size ] = old_f_2[ off ];
            new_f_3_0[ nsd_0.size ] = old_f_3[ off ];
            new_f_4_0[ nsd_0.size ] = old_f_4[ off ];

            new_ids_0[ nsd_0.size ] = old_ids[ off ];

            ++nsd_0.size;
        }
    }
}
