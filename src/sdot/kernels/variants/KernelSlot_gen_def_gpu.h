void mk_items_0_0_1_1_2_2( ShapeData &new_shape_data, const std::array<BI,3> &new_node_indices, const ShapeData &old_shape_data, const std::array<BI,3> &old_node_indices, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 0 ] * dim + 0 ) * new_shape_data.rese;
    TF *new_y_0 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 0 ] * dim + 1 ) * new_shape_data.rese;
    TF *new_x_1 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 1 ] * dim + 0 ) * new_shape_data.rese;
    TF *new_y_1 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 1 ] * dim + 1 ) * new_shape_data.rese;
    TF *new_x_2 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 2 ] * dim + 0 ) * new_shape_data.rese;
    TF *new_y_2 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 2 ] * dim + 1 ) * new_shape_data.rese;

    TI *new_f_0 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 0 ] * new_shape_data.rese;
    TI *new_f_1 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 1 ] * new_shape_data.rese;
    TI *new_f_2 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 2 ] * new_shape_data.rese;

    TI *new_ids = reinterpret_cast<TI *>( new_shape_data.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 0 ] * dim + 0 ) * old_shape_data.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 0 ] * dim + 1 ) * old_shape_data.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 1 ] * dim + 0 ) * old_shape_data.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 1 ] * dim + 1 ) * old_shape_data.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 2 ] * dim + 0 ) * old_shape_data.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 2 ] * dim + 1 ) * old_shape_data.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 0 ] * old_shape_data.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 1 ] * old_shape_data.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 2 ] * old_shape_data.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( old_shape_data.ids );

    const TI *o0 = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::offset_0 ] );
    const TI *o1 = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::offset_1 ] );
    const TI *cc = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::cut_case ] );

    for( BI nmp = 0; nmp < nb_multiprocs(); ++nmp ) {
        for( BI ind = o0[ num_case * nb_multiprocs() + nmp ]; ind < o1[ num_case * nb_multiprocs() + nmp ]; ++ind ) {
            TI off = cc[ ind ];

            new_x_0[ new_shape_data.size ] = old_x_0[ off ];
            new_y_0[ new_shape_data.size ] = old_y_0[ off ];
            new_x_1[ new_shape_data.size ] = old_x_1[ off ];
            new_y_1[ new_shape_data.size ] = old_y_1[ off ];
            new_x_2[ new_shape_data.size ] = old_x_2[ off ];
            new_y_2[ new_shape_data.size ] = old_y_2[ off ];

            new_f_0[ new_shape_data.size ] = old_f_0[ off ];
            new_f_1[ new_shape_data.size ] = old_f_1[ off ];
            new_f_2[ new_shape_data.size ] = old_f_2[ off ];

            new_ids[ new_shape_data.size ] = old_ids[ off ];

            ++new_shape_data.size;
        }
    }
}
void mk_items_0_0_1_1_2_2_3_3( ShapeData &new_shape_data, const std::array<BI,3> &new_node_indices, const ShapeData &old_shape_data, const std::array<BI,3> &old_node_indices, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 0 ] * dim + 0 ) * new_shape_data.rese;
    TF *new_y_0 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 0 ] * dim + 1 ) * new_shape_data.rese;
    TF *new_x_1 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 1 ] * dim + 0 ) * new_shape_data.rese;
    TF *new_y_1 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 1 ] * dim + 1 ) * new_shape_data.rese;
    TF *new_x_2 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 2 ] * dim + 0 ) * new_shape_data.rese;
    TF *new_y_2 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 2 ] * dim + 1 ) * new_shape_data.rese;

    TI *new_f_0 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 0 ] * new_shape_data.rese;
    TI *new_f_1 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 1 ] * new_shape_data.rese;
    TI *new_f_2 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 2 ] * new_shape_data.rese;

    TI *new_ids = reinterpret_cast<TI *>( new_shape_data.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 0 ] * dim + 0 ) * old_shape_data.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 0 ] * dim + 1 ) * old_shape_data.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 1 ] * dim + 0 ) * old_shape_data.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 1 ] * dim + 1 ) * old_shape_data.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 2 ] * dim + 0 ) * old_shape_data.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 2 ] * dim + 1 ) * old_shape_data.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 0 ] * old_shape_data.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 1 ] * old_shape_data.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 2 ] * old_shape_data.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( old_shape_data.ids );

    const TI *o0 = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::offset_0 ] );
    const TI *o1 = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::offset_1 ] );
    const TI *cc = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::cut_case ] );

    for( BI nmp = 0; nmp < nb_multiprocs(); ++nmp ) {
        for( BI ind = o0[ num_case * nb_multiprocs() + nmp ]; ind < o1[ num_case * nb_multiprocs() + nmp ]; ++ind ) {
            TI off = cc[ ind ];

            new_x_0[ new_shape_data.size ] = old_x_0[ off ];
            new_y_0[ new_shape_data.size ] = old_y_0[ off ];
            new_x_1[ new_shape_data.size ] = old_x_1[ off ];
            new_y_1[ new_shape_data.size ] = old_y_1[ off ];
            new_x_2[ new_shape_data.size ] = old_x_2[ off ];
            new_y_2[ new_shape_data.size ] = old_y_2[ off ];

            new_f_0[ new_shape_data.size ] = old_f_0[ off ];
            new_f_1[ new_shape_data.size ] = old_f_1[ off ];
            new_f_2[ new_shape_data.size ] = old_f_2[ off ];

            new_ids[ new_shape_data.size ] = old_ids[ off ];

            ++new_shape_data.size;
        }
    }
}
void mk_items_0_0_1_1_2_2_3_3_4_4( ShapeData &new_shape_data, const std::array<BI,3> &new_node_indices, const ShapeData &old_shape_data, const std::array<BI,3> &old_node_indices, BI num_case, const void *cut_ids, N<2> dim ) override {
    TF *new_x_0 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 0 ] * dim + 0 ) * new_shape_data.rese;
    TF *new_y_0 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 0 ] * dim + 1 ) * new_shape_data.rese;
    TF *new_x_1 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 1 ] * dim + 0 ) * new_shape_data.rese;
    TF *new_y_1 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 1 ] * dim + 1 ) * new_shape_data.rese;
    TF *new_x_2 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 2 ] * dim + 0 ) * new_shape_data.rese;
    TF *new_y_2 = reinterpret_cast<TF *>( new_shape_data.coordinates ) + ( new_node_indices[ 2 ] * dim + 1 ) * new_shape_data.rese;

    TI *new_f_0 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 0 ] * new_shape_data.rese;
    TI *new_f_1 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 1 ] * new_shape_data.rese;
    TI *new_f_2 = reinterpret_cast<TI *>( new_shape_data.face_ids ) + new_node_indices[ 2 ] * new_shape_data.rese;

    TI *new_ids = reinterpret_cast<TI *>( new_shape_data.ids );

    const TF *old_x_0 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 0 ] * dim + 0 ) * old_shape_data.rese;
    const TF *old_y_0 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 0 ] * dim + 1 ) * old_shape_data.rese;
    const TF *old_x_1 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 1 ] * dim + 0 ) * old_shape_data.rese;
    const TF *old_y_1 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 1 ] * dim + 1 ) * old_shape_data.rese;
    const TF *old_x_2 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 2 ] * dim + 0 ) * old_shape_data.rese;
    const TF *old_y_2 = reinterpret_cast<const TF *>( old_shape_data.coordinates ) + ( old_node_indices[ 2 ] * dim + 1 ) * old_shape_data.rese;

    const TI *old_f_0 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 0 ] * old_shape_data.rese;
    const TI *old_f_1 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 1 ] * old_shape_data.rese;
    const TI *old_f_2 = reinterpret_cast<const TI *>( old_shape_data.face_ids ) + old_node_indices[ 2 ] * old_shape_data.rese;

    const TI *old_ids = reinterpret_cast<const TI *>( old_shape_data.ids );

    const TI *o0 = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::offset_0 ] );
    const TI *o1 = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::offset_1 ] );
    const TI *cc = reinterpret_cast<const TI *>( old_shape_data.tmp[ ShapeData::cut_case ] );

    for( BI nmp = 0; nmp < nb_multiprocs(); ++nmp ) {
        for( BI ind = o0[ num_case * nb_multiprocs() + nmp ]; ind < o1[ num_case * nb_multiprocs() + nmp ]; ++ind ) {
            TI off = cc[ ind ];

            new_x_0[ new_shape_data.size ] = old_x_0[ off ];
            new_y_0[ new_shape_data.size ] = old_y_0[ off ];
            new_x_1[ new_shape_data.size ] = old_x_1[ off ];
            new_y_1[ new_shape_data.size ] = old_y_1[ off ];
            new_x_2[ new_shape_data.size ] = old_x_2[ off ];
            new_y_2[ new_shape_data.size ] = old_y_2[ off ];

            new_f_0[ new_shape_data.size ] = old_f_0[ off ];
            new_f_1[ new_shape_data.size ] = old_f_1[ off ];
            new_f_2[ new_shape_data.size ] = old_f_2[ off ];

            new_ids[ new_shape_data.size ] = old_ids[ off ];

            ++new_shape_data.size;
        }
    }
}
