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

        VI ids = VI::load_aligned( id_data + beg ) * SimdSize<TF,Arch>::value + VI::iota();
        VF old = VF::gather( tmp_f.data(), ids );

        VF p_0_0 = VF::load_aligned( p_0_0_ptr + beg );
        VF p_0_1 = VF::load_aligned( p_0_1_ptr + beg );
        VF p_1_0 = VF::load_aligned( p_1_0_ptr + beg );
        VF p_1_1 = VF::load_aligned( p_1_1_ptr + beg );
        VF p_2_0 = VF::load_aligned( p_2_0_ptr + beg );
        VF p_2_1 = VF::load_aligned( p_2_1_ptr + beg );

        VF R0 = 0;
        VF R1 = p_1_1;
        VF R2 = p_0_1;
        VF R3 = R1 - R2;
        VF R4 = 1;
        VF R5 = p_0_0;
        VF R6 = R5 - R5;
        VF R7 = R4 * R6;
        VF R8 = R3 * R7;
        VF R9 = R0 + R8;
        VF R10 = R2 - R2;
        VF R11 = p_1_0;
        VF R12 = R11 - R5;
        VF R13 = -1;
        VF R14 = R12 * R13;
        VF R15 = R10 * R14;
        VF R16 = R9 + R15;
        VF R17 = R0 + R16;
        VF R18 = p_2_1;
        VF R19 = R2 - R18;
        VF R20 = p_2_0;
        VF R21 = R20 - R5;
        VF R22 = R4 * R21;
        VF R23 = R19 * R22;
        VF R24 = R0 + R23;
        VF R25 = R18 - R2;
        VF R26 = R5 - R20;
        VF R27 = R26 * R13;
        VF R28 = R25 * R27;
        VF R29 = R24 + R28;
        VF R30 = R17 + R29;
        VF R31 = R18 - R1;
        VF R32 = R12 * R4;
        VF R33 = R31 * R32;
        VF R34 = R0 + R33;
        VF R35 = R20 - R11;
        VF R36 = R35 * R13;
        VF R37 = R3 * R36;
        VF R38 = R34 + R37;
        VF R39 = R30 + R38;
        VF R40 = R0 + R39;
        VF R41 = 2;
        VF R42 = R40 / R41;
        VF res = R42;

        VF::scatter( tmp_f.data(), ids, old + res );
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

        VI ids = VI::load_aligned( id_data + beg ) * SimdSize<TF,Arch>::value + VI::iota();
        VF old = VF::gather( tmp_f.data(), ids );

        VF p_0_0 = VF::load_aligned( p_0_0_ptr + beg );
        VF p_0_1 = VF::load_aligned( p_0_1_ptr + beg );
        VF p_1_0 = VF::load_aligned( p_1_0_ptr + beg );
        VF p_1_1 = VF::load_aligned( p_1_1_ptr + beg );
        VF p_2_0 = VF::load_aligned( p_2_0_ptr + beg );
        VF p_2_1 = VF::load_aligned( p_2_1_ptr + beg );
        VF p_3_0 = VF::load_aligned( p_3_0_ptr + beg );
        VF p_3_1 = VF::load_aligned( p_3_1_ptr + beg );

        VF R0 = 0;
        VF R1 = p_1_1;
        VF R2 = p_0_1;
        VF R3 = R1 - R2;
        VF R4 = 1;
        VF R5 = p_0_0;
        VF R6 = R5 - R5;
        VF R7 = R4 * R6;
        VF R8 = R3 * R7;
        VF R9 = R0 + R8;
        VF R10 = R2 - R2;
        VF R11 = p_1_0;
        VF R12 = R11 - R5;
        VF R13 = -1;
        VF R14 = R12 * R13;
        VF R15 = R10 * R14;
        VF R16 = R9 + R15;
        VF R17 = R0 + R16;
        VF R18 = p_3_1;
        VF R19 = R2 - R18;
        VF R20 = p_3_0;
        VF R21 = R20 - R5;
        VF R22 = R4 * R21;
        VF R23 = R19 * R22;
        VF R24 = R0 + R23;
        VF R25 = R18 - R2;
        VF R26 = R5 - R20;
        VF R27 = R26 * R13;
        VF R28 = R25 * R27;
        VF R29 = R24 + R28;
        VF R30 = R17 + R29;
        VF R31 = p_2_1;
        VF R32 = R31 - R1;
        VF R33 = R12 * R4;
        VF R34 = R32 * R33;
        VF R35 = R0 + R34;
        VF R36 = p_2_0;
        VF R37 = R36 - R11;
        VF R38 = R37 * R13;
        VF R39 = R3 * R38;
        VF R40 = R35 + R39;
        VF R41 = R30 + R40;
        VF R42 = R18 - R31;
        VF R43 = R36 - R5;
        VF R44 = R4 * R43;
        VF R45 = R42 * R44;
        VF R46 = R0 + R45;
        VF R47 = R31 - R2;
        VF R48 = R20 - R36;
        VF R49 = R48 * R13;
        VF R50 = R47 * R49;
        VF R51 = R46 + R50;
        VF R52 = R41 + R51;
        VF R53 = R0 + R52;
        VF R54 = 2;
        VF R55 = R53 / R54;
        VF res = R55;

        VF::scatter( tmp_f.data(), ids, old + res );
    } );
    continue;
}
