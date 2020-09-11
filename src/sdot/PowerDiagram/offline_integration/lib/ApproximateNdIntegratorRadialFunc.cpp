#include "ApproximateNdIntegratorRadialFunc.h"
#include "../../system/Assert.h"
//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O3

ApproximateNdIntegratorRadialFunc::ApproximateNdIntegratorRadialFunc( TF epsilon, int degp ) :epsilon( epsilon ), degp( degp ) {
    continuity = 1;
}

void ApproximateNdIntegratorRadialFunc::run( const std::vector<TF> &r, const std::vector<TF> &f ) {
    using std::pow;

    // get a polynomial approximation of the radial func f( r )
    // (needed to be able to get a correct approximation of the integration of r * f( r ))
    PolyApproximator pa_f;
    pa_f.continuity = 0;
    pa_f.epsilon = epsilon / 10;
    pa_f.degp = 7; ///< we're using high precision number, so that it's legitimate to use high degree polynomials

    pa_f.run( r, f );

    // integrate( r * f( r ), r = 0 .. s^0.5 ) / s using approximation of f
    // the result is stored in a set of points
    TF acc = 0;
    std::vector<TF> integ_ss;
    std::vector<TF> integ_ys;
    for( std::size_t num_cut = 0; num_cut < pa_f.cuts.size(); ++num_cut ) {
        TF beg = num_cut ? pa_f.cuts[ num_cut - 1 ].position : 0;
        TF end = pa_f.cuts[ num_cut - 0 ].position;

        TF step = ( end - beg ) / ( TF( 10 ) * r.size() );
        const PolyApproximator::Cut &cut = pa_f.cuts[ num_cut ];
        for( TF pos = beg + 0.5 * step; pos < end; pos += step ) {
            TF v = acc;
            for( std::size_t i = 0; i < cut.coeffs.size(); ++i )
                v += cut.coeffs[ i ] / ( i + 2 ) * ( pow( pos, i + 2 ) - pow( beg, i + 2 ) );
            integ_ss.push_back( pow( pos, 2 ) );
            integ_ys.push_back( v / pow( pos, 2 ) );
        }
        for( std::size_t i = 0; i < cut.coeffs.size(); ++i )
            acc += cut.coeffs[ i ] / ( i + 2 ) * ( pow( end, i + 2 ) - pow( beg, i + 2 ) );
    }

    // approximation of the integral
    coeffs.continuity = continuity;
    coeffs.epsilon = epsilon;
    coeffs.degp = degp;

    coeffs.run( integ_ss, integ_ys );

    // computation of the real error
    TF error = 0;
    for( std::size_t i = 100; i < integ_ss.size(); ++i )
        error = max( error, abs( integ_ys[ i ] - coeffs.apply( integ_ss[ i ] ) ) );

    P( coeffs.cuts.size(), error );
}

void ApproximateNdIntegratorRadialFunc::run( const std::function<TF(TF)> &func, TF beg_of_log_scale, std::size_t nb_values_before_log_scale, TF end_of_log_scale, std::size_t nb_values_after_log_scale ) {
    using std::pow;

    std::vector<TF> x( nb_values_before_log_scale + nb_values_after_log_scale );
    std::vector<TF> y( nb_values_before_log_scale + nb_values_after_log_scale );
    for( size_t i = 0; i < nb_values_before_log_scale; ++i ) {
        x[ i ] = i * beg_of_log_scale / nb_values_before_log_scale;
        y[ i ] = func( x[ i ] );
    }

    for( size_t i = 0; i < nb_values_after_log_scale; ++i ) {
        x[ nb_values_before_log_scale + i ] = beg_of_log_scale * pow( end_of_log_scale / beg_of_log_scale, TF( i ) / ( nb_values_after_log_scale - 1 ) );
        y[ nb_values_before_log_scale + i ] = func( x[ nb_values_before_log_scale + i ] );
    }

    run( x, y );
}

void ApproximateNdIntegratorRadialFunc::write_to_stream( std::ostream &os ) const {
    os << "coeffs := [";
    for( std::size_t i = 0; i < coeffs.cuts.size(); ++i )
        os << ( i ? ", " : "" ) << "\n    " << coeffs.cuts[ i ];
    os << "\n]" << std::endl;
}

ApproximateNdIntegratorRadialFunc::TF ApproximateNdIntegratorRadialFunc::integrate( const std::vector<std::array<TF,2>> &points ) const {
    using std::pow;

    auto cut_index_r = [&]( TF r2 ) {
        std::size_t beg = 0;
        std::size_t end = coeffs.cuts.size() - 1;
        while ( beg < end ) {
            std::size_t mid = beg + ( end - beg ) / 2;
            if ( coeffs.cuts[ mid ].position < r2 )
                beg = mid + 1;
            else
                end = mid;
        }
        return beg;
    };

    auto part_int = [&]( std::array<TF,2> P0, std::array<TF,2> P1, TF u0, TF u1, std::size_t index ) {
        const auto &cuts = coeffs.cuts[ index ];
        TF R_0 = cuts.coeffs[ 1 ]; TF R_1 = cuts.coeffs[ 0 ]; TF R_2 = cuts.coeffs[ 2 ]; TF R_3 = 2.0*R_2;
        TF R_4 = cuts.coeffs[ 3 ]; TF R_5 = 3.0*R_4; TF R_6 = cuts.coeffs[ 4 ]; TF R_7 = 4.0*R_6;
        TF R_8 = cuts.coeffs[ 5 ]; TF R_9 = 5.0*R_8; TF R_10 = cuts.coeffs[ 7 ]; TF R_11 = cuts.coeffs[ 6 ];
        TF R_12 = 23040.0*R_11; TF R_13 = 2304.0*R_11; TF R_14 = 288.0*R_11; TF R_15 = 6.0*R_11;
        TF R_16 = 12.0*R_11; TF R_17 = 48.0*R_11; TF R_18 = P0[ 1 ]; TF R_19 = (-1.0)*R_18;
        TF R_20 = P1[ 1 ]; TF R_21 = (-1.0)*R_20; R_21 = R_21+R_18; R_19 = R_20+R_19;
        TF R_22 = pow(R_19,2); TF R_23 = P1[ 0 ]; TF R_24 = (-1.0)*R_23; TF R_25 = P0[ 0 ];
        TF R_26 = (-1.0)*R_25; R_26 = R_23+R_26; TF R_27 = pow(R_26,2); R_22 = R_27+R_22;
        R_27 = pow(R_22,6); TF R_28 = pow(R_22,4); TF R_29 = pow(R_22,2); TF R_30 = R_21*R_26;
        R_30 = (-1.0)*R_30; R_24 = R_25+R_24; TF R_31 = R_24*R_19; R_30 = R_31+R_30;
        R_31 = u0; TF R_32 = (-1.0)*R_31; TF R_33 = u1; R_31 = R_31+R_33;
        R_20 = R_20*R_31; R_20 = 0.5*R_20; R_23 = R_23*R_31; R_23 = 0.5*R_23;
        R_31 = -0.5*R_31; R_31 = 1.0+R_31; R_18 = R_18*R_31; R_18 = R_20+R_18;
        R_24 = R_24*R_18; R_20 = pow(R_18,2); R_18 = R_19*R_18; R_31 = R_25*R_31;
        R_23 = R_31+R_23; R_21 = R_21*R_23; R_21 = (-1.0)*R_21; R_24 = R_21+R_24;
        R_21 = R_22*R_24; R_31 = 87178291200.0*R_21; R_25 = 3.0*R_21; R_19 = pow(R_23,2);
        R_20 = R_19+R_20; R_19 = R_10*R_20; TF R_34 = 161280.0*R_19; R_34 = R_12+R_34;
        R_12 = R_22*R_34; TF R_35 = 10395.0*R_12; TF R_36 = 1155.0*R_12; TF R_37 = 3465.0*R_12;
        TF R_38 = 165.0*R_12; R_12 = 11.0*R_12; TF R_39 = 13440.0*R_19; R_39 = R_13+R_39;
        R_39 = R_20*R_39; R_11 = R_11+R_19; R_11 = R_20*R_11; R_11 = R_8+R_11;
        R_11 = R_20*R_11; R_11 = R_6+R_11; R_11 = R_20*R_11; R_11 = R_4+R_11;
        R_11 = R_20*R_11; R_11 = R_2+R_11; R_11 = R_20*R_11; R_11 = R_0+R_11;
        R_11 = R_20*R_11; R_11 = R_1+R_11; R_11 = R_24*R_11; R_1 = 1344.0*R_19;
        R_1 = R_14+R_1; R_1 = R_20*R_1; R_14 = 7.0*R_19; R_14 = R_15+R_14;
        R_14 = R_20*R_14; R_14 = R_9+R_14; R_9 = R_20*R_14; R_9 = R_7+R_9;
        R_7 = R_20*R_9; R_7 = R_5+R_7; R_5 = R_20*R_7; R_5 = R_3+R_5;
        R_3 = R_20*R_5; R_3 = R_0+R_3; R_0 = R_22*R_3; R_5 = 2.0*R_5;
        R_7 = 2.0*R_7; R_9 = 2.0*R_9; R_14 = 2.0*R_14; R_15 = 28.0*R_19;
        R_15 = R_16+R_15; R_15 = R_20*R_15; R_14 = R_15+R_14; R_15 = R_20*R_14;
        R_9 = R_15+R_9; R_15 = R_20*R_9; R_7 = R_15+R_7; R_15 = R_20*R_7;
        R_5 = R_15+R_5; R_15 = R_22*R_5; R_15 = 3.0*R_15; R_7 = 4.0*R_7;
        R_9 = 4.0*R_9; R_14 = 4.0*R_14; R_19 = 168.0*R_19; R_19 = R_17+R_19;
        R_19 = R_20*R_19; R_14 = R_19+R_14; R_19 = 6.0*R_14; R_19 = R_1+R_19;
        R_1 = 8.0*R_19; R_1 = R_39+R_1; R_39 = R_22*R_1; R_39 = 9.0*R_39;
        R_19 = R_20*R_19; R_14 = R_20*R_14; R_9 = R_14+R_9; R_14 = 6.0*R_9;
        R_14 = R_19+R_14; R_19 = R_22*R_14; R_19 = 7.0*R_19; R_9 = R_20*R_9;
        R_7 = R_9+R_7; R_9 = R_22*R_7; R_9 = 5.0*R_9; R_23 = R_26*R_23;
        R_18 = R_23+R_18; R_23 = R_10*R_18; R_26 = R_30*R_18; R_20 = 1220496076800.0*R_26;
        R_20 = R_31+R_20; R_20 = R_27*R_20; R_20 = R_10*R_20; R_25 = R_26+R_25;
        R_25 = R_18*R_25; R_25 = R_10*R_25; R_25 = 2905943040.0*R_25; R_27 = R_26+R_21;
        R_23 = R_27*R_23; R_23 = 2905943040.0*R_23; R_31 = 2.0*R_26; R_21 = R_31+R_21;
        R_3 = R_18*R_3; R_3 = R_30*R_3; R_3 = 2.0*R_3; R_31 = pow(R_18,2);
        R_17 = R_24*R_31; R_17 = R_22*R_17; R_10 = R_10*R_31; R_16 = 5588352000.0*R_10;
        R_16 = R_35+R_16; R_16 = R_26*R_16; R_16 = 16.0*R_16; R_26 = 2682408960.0*R_10;
        R_26 = R_35+R_26; R_26 = R_30*R_26; R_26 = R_23+R_26; R_26 = 3.0*R_26;
        R_25 = R_26+R_25; R_25 = R_18*R_25; R_26 = 121927680.0*R_10; R_26 = R_36+R_26;
        R_26 = R_31*R_26; R_36 = 894136320.0*R_10; R_36 = R_37+R_36; R_27 = R_36*R_27;
        R_27 = 3.0*R_27; R_25 = R_27+R_25; R_25 = 2.0*R_25; R_16 = R_25+R_16;
        R_16 = R_28*R_16; R_36 = R_17*R_36; R_17 = 9031680.0*R_10; R_38 = R_17+R_38;
        R_38 = R_31*R_38; R_10 = 322560.0*R_10; R_12 = R_10+R_12; R_12 = R_31*R_12;
        R_34 = R_31*R_34; R_39 = R_34+R_39; R_34 = R_22*R_39; R_10 = 105.0*R_34;
        R_26 = R_10+R_26; R_10 = R_30*R_26; R_10 = R_18*R_10; R_10 = 16.0*R_10;
        R_21 = R_26*R_21; R_21 = R_36+R_21; R_21 = 2.0*R_21; R_10 = R_21+R_10;
        R_10 = R_29*R_10; R_29 = 42.0*R_34; R_38 = R_29+R_38; R_38 = R_31*R_38;
        R_34 = 7.0*R_34; R_34 = R_12+R_34; R_34 = R_31*R_34; R_39 = R_31*R_39;
        R_1 = R_31*R_1; R_19 = R_1+R_19; R_1 = R_22*R_19; R_1 = 5.0*R_1;
        R_1 = R_39+R_1; R_39 = R_22*R_1; R_39 = 3.0*R_39; R_38 = R_39+R_38;
        R_38 = R_22*R_38; R_38 = R_24*R_38; R_38 = (1.0/90720.0)*R_38; R_39 = R_34+R_39;
        R_39 = R_18*R_39; R_39 = R_30*R_39; R_39 = (1.0/11340.0)*R_39; R_38 = R_39+R_38;
        R_1 = R_31*R_1; R_19 = R_31*R_19; R_5 = R_5*R_31; R_0 = R_5+R_0;
        R_0 = R_24*R_0; R_3 = R_0+R_3; R_14 = R_31*R_14; R_9 = R_14+R_9;
        R_14 = R_22*R_9; R_14 = 3.0*R_14; R_14 = R_19+R_14; R_19 = R_22*R_14;
        R_19 = R_1+R_19; R_19 = R_24*R_19; R_19 = (1.0/1260.0)*R_19; R_14 = R_18*R_14;
        R_14 = R_30*R_14; R_14 = (1.0/210.0)*R_14; R_19 = R_14+R_19; R_9 = R_31*R_9;
        R_7 = R_31*R_7; R_15 = R_7+R_15; R_22 = R_22*R_15; R_22 = R_9+R_22;
        R_22 = R_24*R_22; R_22 = (1.0/30.0)*R_22; R_15 = R_18*R_15; R_15 = R_30*R_15;
        R_15 = (2.0/15.0)*R_15; R_22 = R_15+R_22; R_32 = R_33+R_32; R_33 = pow(R_32,15);
        R_33 = R_20*R_33; R_33 = (1.0/21424936845312000.0)*R_33; R_20 = pow(R_32,13); R_20 = R_16*R_20;
        R_20 = (1.0/25505877196800.0)*R_20; R_16 = pow(R_32,11); R_16 = R_10*R_16; R_16 = (1.0/40874803200.0)*R_16;
        R_10 = pow(R_32,9); R_38 = R_10*R_38; R_38 = (1.0/512.0)*R_38; R_10 = pow(R_32,7);
        R_19 = R_10*R_19; R_19 = (1.0/128.0)*R_19; R_10 = pow(R_32,3); R_10 = R_3*R_10;
        R_10 = (1.0/12.0)*R_10; R_11 = R_11*R_32; R_10 = R_11+R_10; R_32 = pow(R_32,5);
        R_22 = R_32*R_22; R_22 = (1.0/32.0)*R_22; R_10 = R_22+R_10; R_19 = R_10+R_19;
        R_38 = R_19+R_38; R_16 = R_38+R_16; R_20 = R_16+R_20; R_33 = R_20+R_33;
        return R_33;
    };

    auto int_line = [&]( std::array<TF,2> P0, std::array<TF,2> P1 ) {
        std::size_t index = cut_index_r( pow( P0[ 0 ], 2 ) + pow( P0[ 1 ], 2 ) );
        TF result = 0;
        for( TF u = 0; ; ) {
            std::size_t new_index = index;
            TF new_u = 1;

            // test if line is going to cut a circle at a lower index
            if ( index ) {
                TF d = coeffs.cuts[ index - 1 ].position; d = - d; TF b = P0[ 1 ]; TF R_2 = pow( b, 2 );
                TF a = (-1.0)*b; TF R_4 = P1[ 1 ]; a += R_4; b *= a;
                a *= a; R_4 = P0[ 0 ]; TF R_5 = pow(R_4,2); R_2 += R_5;
                d = R_2+d; R_2 = (-1.0)*R_4; R_5 = P1[ 0 ];
                R_2 = R_5+R_2; R_4 = R_4*R_2; b += R_4; R_4 = pow(b,2);
                b = (-1.0)*b; R_2 = pow(R_2,2); a = R_2+a;
                d = R_4 - a * d;
                if ( d > 0 ) {
                    TF prop_u = ( b - sqrt( d ) ) / a;
                    if ( prop_u > u && prop_u < new_u ) {
                        new_index = index - 1;
                        new_u = prop_u;
                    }
                    prop_u = ( b + sqrt( d ) ) / a;
                    if ( prop_u > u && prop_u < new_u ) {
                        new_index = index - 1;
                        new_u = prop_u;
                    }
                }
            }

            // test if line is going to cut a circle at an higher index
            if ( index < coeffs.cuts.size() - 1 ) {
                TF a; TF b; TF c; TF d;

                TF R_0 = coeffs.cuts[ index ].position; R_0 = (-1.0)*R_0; TF R_1 = P0[ 1 ]; TF R_2 = pow(R_1,2);
                TF R_3 = (-1.0)*R_1; TF R_4 = P1[ 1 ]; R_3 = R_4+R_3; R_1 = R_1*R_3;
                R_3 = pow(R_3,2); R_4 = P0[ 0 ]; TF R_5 = pow(R_4,2); R_2 = R_5+R_2;
                R_0 = R_2+R_0; c = R_0; R_2 = (-1.0)*R_4; R_5 = P1[ 0 ];
                R_2 = R_5+R_2; R_4 = R_4*R_2; R_1 = R_4+R_1; R_4 = pow(R_1,2);
                R_1 = (-1.0)*R_1; b = R_1; R_2 = pow(R_2,2); R_3 = R_2+R_3;
                a = R_3; R_0 = R_3*R_0; R_0 = (-1.0)*R_0; R_0 = R_4+R_0;
                d = R_0;
                if ( d > 0 ) {
                    TF prop_u = ( b - sqrt( d ) ) / a;
                    if ( prop_u > u && prop_u < new_u ) {
                        new_index = index + 1;
                        new_u = prop_u;
                    }
                    prop_u = ( b + sqrt( d ) ) / a;
                    if ( prop_u > u && prop_u < new_u ) {
                        new_index = index + 1;
                        new_u = prop_u;
                    }
                }
            }

            // integration on sub part of the line
            result += part_int( P0, P1, u, new_u, index );

            // next disc
            if ( new_u >= 1 )
                break;
            index = new_index;
            u = new_u;
        }
        return result;
    };

    TF res = 0;
    for( std::size_t i1 = 0, i0 = points.size() - 1; i1 < points.size(); i0 = i1++ )
        res += int_line( points[ i0 ], points[ i1 ] );
    return res;
}

void ApproximateNdIntegratorRadialFunc::generate( std::ostream &os ) const {
    auto only_one_coeff = [&]( int &coeff_index, TF &coeff_val, int num_cut ) {
        coeff_index = -1;
        for( std::size_t i = 0; i < coeffs.cuts[ num_cut ].coeffs.size(); ++i ) {
            if ( coeffs.cuts[ num_cut ].coeffs[ i ] ) {
                if ( coeff_index >= 0 )
                    return false;
                coeff_val = coeffs.cuts[ num_cut ].coeffs[ i ];
                coeff_index = i;
            }
        }
        return true;
    };

    if ( coeffs.cuts.size() == 1 ) {
        os << "auto seg_val = []( PT P0, PT P1 ) {\n";
        int coeff_index = 0;
        TF coeff_val;
        if ( only_one_coeff( coeff_index, coeff_val, 0 ) ) {
            // generated using `metil src/PowerDiagram/offline_integration/lib/gen.met `
            switch ( coeff_index ) {
            case 0:
                os << "    TF result; \n";
                os << "    TF R_0 = P1.x; TF R_1 = (-1.0)*R_0; TF R_2 = P0.x; R_1 = R_1+R_2;\n";
                os << "    R_0 = R_2+R_0; R_2 = P0.y; TF R_3 = P1.y; TF R_4 = R_2+R_3;\n";
                os << "    R_1 = R_4*R_1; R_1 = (-1.0)*R_1; R_3 = (-1.0)*R_3; R_2 = R_3+R_2;\n";
                os << "    R_0 = R_2*R_0; R_1 = R_0+R_1; R_0 = " << coeff_val << "; R_1 = R_0*R_1;\n";
                os << "    R_1 = -0.5*R_1; result = R_1; \n";
                os << "    return result;\n";
                break;
            case 1:
                os << "    TF result; \n";
                os << "    TF R_0 = P1.y; TF R_1 = (-1.0)*R_0; TF R_2 = P0.y; TF R_3 = (-1.0)*R_2;\n";
                os << "    R_3 = R_0+R_3; TF R_4 = pow(R_3,2); R_1 = R_2+R_1; R_0 = R_2+R_0;\n";
                os << "    R_2 = R_3*R_0; TF R_5 = pow(R_0,2); TF R_6 = P0.x; TF R_7 = (-1.0)*R_6;\n";
                os << "    TF R_8 = P1.x; R_7 = R_8+R_7; TF R_9 = R_1*R_7; R_9 = (-1.0)*R_9;\n";
                os << "    TF R_10 = pow(R_7,2); R_4 = R_10+R_4; R_10 = (-1.0)*R_8; R_10 = R_6+R_10;\n";
                os << "    R_3 = R_10*R_3; R_3 = R_9+R_3; R_0 = R_10*R_0; R_0 = (-1.0)*R_0;\n";
                os << "    R_6 = R_8+R_6; R_7 = R_7*R_6; R_7 = R_2+R_7; R_7 = R_3*R_7;\n";
                os << "    R_7 = 2.0*R_7; R_1 = R_1*R_6; R_0 = R_1+R_0; R_4 = R_4*R_0;\n";
                os << "    R_4 = (-1.0)*R_4; R_7 = R_4+R_7; R_6 = pow(R_6,2); R_5 = R_6+R_5;\n";
                os << "    R_6 = " << coeff_val << "; R_7 = R_6*R_7; R_7 = (1.0/24.0)*R_7; R_5 = R_6*R_5;\n";
                os << "    R_0 = R_5*R_0; R_0 = -0.125*R_0; R_7 = R_0+R_7; result = R_7;\n";
                os << "    return result;\n";
                break;
            case 2:
                os << "    TF result; \n";
                os << "    TF R_0 = P0.y; TF R_1 = (-1.0)*R_0; TF R_2 = P1.y; TF R_3 = R_2+R_0;\n";
                os << "    TF R_4 = pow(R_3,2); TF R_5 = (-1.0)*R_2; R_5 = R_0+R_5; R_1 = R_2+R_1;\n";
                os << "    R_2 = R_1*R_3; R_0 = pow(R_1,2); TF R_6 = P0.x; TF R_7 = (-1.0)*R_6;\n";
                os << "    TF R_8 = P1.x; TF R_9 = R_8+R_6; TF R_10 = pow(R_9,2); R_4 = R_10+R_4;\n";
                os << "    R_10 = pow(R_4,2); TF R_11 = R_5*R_9; TF R_12 = (-1.0)*R_8; R_6 = R_12+R_6;\n";
                os << "    R_3 = R_6*R_3; R_3 = (-1.0)*R_3; R_3 = R_11+R_3; R_1 = R_6*R_1;\n";
                os << "    R_7 = R_8+R_7; R_9 = R_7*R_9; R_9 = R_2+R_9; R_2 = pow(R_9,2);\n";
                os << "    R_2 = 0.5*R_2; R_5 = R_5*R_7; R_5 = (-1.0)*R_5; R_1 = R_5+R_1;\n";
                os << "    R_5 = R_1*R_4; R_5 = R_9*R_5; R_9 = R_1*R_9; R_9 = 48.0*R_9;\n";
                os << "    R_7 = pow(R_7,2); R_0 = R_7+R_0; R_4 = R_0*R_4; R_4 = 0.25*R_4;\n";
                os << "    R_2 = R_4+R_2; R_2 = R_3*R_2; R_2 = (-2.0)*R_2; R_2 = R_5+R_2;\n";
                os << "    R_5 = R_0*R_3; R_5 = (-12.0)*R_5; R_5 = R_9+R_5; R_5 = R_0*R_5;\n";
                os << "    R_0 = " << coeff_val << "; R_2 = R_0*R_2; R_2 = (1.0/24.0)*R_2; R_10 = R_0*R_10;\n";
                os << "    R_3 = R_10*R_3; R_3 = (-1.0/32.0)*R_3; R_2 = R_3+R_2; R_5 = R_0*R_5;\n";
                os << "    R_5 = (1.0/1920.0)*R_5; R_2 = R_5+R_2; result = R_2; \n";
                os << "    return result;\n";
                break;
            case 3:
                os << "    TF result; \n";
                os << "    TF R_0 = P0.y; TF R_1 = (-1.0)*R_0; TF R_2 = P1.y; TF R_3 = R_2+R_0;\n";
                os << "    TF R_4 = pow(R_3,2); TF R_5 = (-1.0)*R_2; R_5 = R_0+R_5; R_1 = R_2+R_1;\n";
                os << "    R_2 = R_1*R_3; R_0 = pow(R_1,2); TF R_6 = P0.x; TF R_7 = (-1.0)*R_6;\n";
                os << "    TF R_8 = P1.x; TF R_9 = (-1.0)*R_8; R_9 = R_6+R_9; R_1 = R_9*R_1;\n";
                os << "    R_3 = R_9*R_3; R_3 = (-1.0)*R_3; R_6 = R_6+R_8; R_9 = pow(R_6,2);\n";
                os << "    R_4 = R_9+R_4; R_9 = pow(R_4,3); TF R_10 = R_5*R_6; R_3 = R_10+R_3;\n";
                os << "    R_7 = R_8+R_7; R_6 = R_7*R_6; R_6 = R_2+R_6; R_2 = pow(R_6,2);\n";
                os << "    R_8 = 2.0*R_2; R_10 = 1.5*R_2; R_5 = R_5*R_7; R_5 = (-1.0)*R_5;\n";
                os << "    R_1 = R_5+R_1; R_5 = R_1*R_6; TF R_11 = R_4*R_5; R_11 = 0.375*R_11;\n";
                os << "    TF R_12 = 14.0*R_5; TF R_13 = 2160.0*R_5; R_7 = pow(R_7,2); R_0 = R_7+R_0;\n";
                os << "    R_7 = R_0*R_4; TF R_14 = 3.0*R_7; R_14 = R_8+R_14; R_14 = R_5*R_14;\n";
                os << "    R_14 = 3.0*R_14; R_5 = 0.25*R_7; R_5 = R_2+R_5; R_5 = R_3*R_5;\n";
                os << "    R_2 = -0.75*R_5; R_2 = R_11+R_2; R_2 = R_4*R_2; R_5 = (-1.0)*R_5;\n";
                os << "    R_7 = 0.5*R_7; R_10 = R_7+R_10; R_10 = R_1*R_10; R_1 = R_0*R_6;\n";
                os << "    R_1 = R_3*R_1; R_1 = -2.5*R_1; R_10 = R_1+R_10; R_10 = R_6*R_10;\n";
                os << "    R_10 = 2.0*R_10; R_6 = R_0*R_3; R_1 = (-5.0)*R_6; R_12 = R_1+R_12;\n";
                os << "    R_12 = R_4*R_12; R_12 = 0.25*R_12; R_12 = R_5+R_12; R_12 = R_0*R_12;\n";
                os << "    R_12 = R_10+R_12; R_12 = 6.0*R_12; R_14 = R_12+R_14; R_6 = (-360.0)*R_6;\n";
                os << "    R_13 = R_6+R_13; R_0 = pow(R_0,2); R_13 = R_0*R_13; R_0 = " << coeff_val << ";\n";
                os << "    R_2 = R_0*R_2; R_2 = (1.0/24.0)*R_2; R_9 = R_0*R_9; R_3 = R_9*R_3;\n";
                os << "    R_3 = (-1.0/128.0)*R_3; R_2 = R_3+R_2; R_14 = R_0*R_14; R_14 = (1.0/1920.0)*R_14;\n";
                os << "    R_2 = R_14+R_2; R_13 = R_0*R_13; R_13 = (1.0/322560.0)*R_13; R_2 = R_13+R_2;\n";
                os << "    result = R_2; \n";
                os << "    return result;\n";
                break;
            case 4:
                os << "    TF result; \n";
                os << "    TF R_0 = P0.x; TF R_1 = (-1.0)*R_0; TF R_2 = P1.x; TF R_3 = (-1.0)*R_2;\n";
                os << "    R_3 = R_0+R_3; R_0 = R_0+R_2; TF R_4 = pow(R_0,2); R_1 = R_2+R_1;\n";
                os << "    R_2 = R_1*R_0; TF R_5 = pow(R_1,2); TF R_6 = P0.y; TF R_7 = (-1.0)*R_6;\n";
                os << "    TF R_8 = P1.y; TF R_9 = R_6+R_8; TF R_10 = pow(R_9,2); R_10 = R_4+R_10;\n";
                os << "    R_4 = pow(R_10,2); TF R_11 = pow(R_10,4); TF R_12 = R_3*R_9; R_12 = (-1.0)*R_12;\n";
                os << "    TF R_13 = (-1.0)*R_8; R_13 = R_6+R_13; R_1 = R_13*R_1; R_1 = (-1.0)*R_1;\n";
                os << "    R_0 = R_13*R_0; R_12 = R_0+R_12; R_7 = R_8+R_7; R_9 = R_7*R_9;\n";
                os << "    R_9 = R_2+R_9; R_2 = R_10*R_9; R_8 = pow(R_9,2); R_0 = 6.0*R_8;\n";
                os << "    R_13 = 480.0*R_8; R_6 = 1.5*R_8; TF R_14 = 2.0*R_8; R_8 = 20.0*R_8;\n";
                os << "    R_3 = R_3*R_7; R_1 = R_3+R_1; R_3 = R_1*R_10; R_3 = R_9*R_3;\n";
                os << "    TF R_15 = 0.125*R_3; R_2 = R_1*R_2; R_2 = 0.125*R_2; TF R_16 = R_1*R_9;\n";
                os << "    TF R_17 = 4.75*R_16; TF R_18 = 546.0*R_16; TF R_19 = 136.0*R_16; R_16 = 161280.0*R_16;\n";
                os << "    R_7 = pow(R_7,2); R_5 = R_7+R_5; R_7 = R_5*R_9; R_7 = R_12*R_7;\n";
                os << "    R_7 = -3.5*R_7; TF R_20 = R_5*R_10; TF R_21 = 4.5*R_20; R_21 = R_0+R_21;\n";
                os << "    R_21 = R_3*R_21; R_3 = 360.0*R_20; R_3 = R_13+R_3; R_3 = R_1*R_3;\n";
                os << "    R_3 = R_9*R_3; R_13 = 0.25*R_20; R_13 = R_6+R_13; R_13 = R_12*R_13;\n";
                os << "    R_6 = -0.25*R_13; R_15 = R_6+R_15; R_15 = R_4*R_15; R_4 = -0.5*R_13;\n";
                os << "    R_2 = R_4+R_2; R_2 = R_9*R_2; R_2 = 2.0*R_2; R_13 = (-2.0)*R_13;\n";
                os << "    R_4 = 0.5*R_20; R_14 = R_4+R_14; R_14 = R_1*R_14; R_7 = R_14+R_7;\n";
                os << "    R_14 = R_9*R_7; R_14 = 3.0*R_14; R_4 = R_10*R_7; R_4 = 0.25*R_4;\n";
                os << "    R_2 = R_4+R_2; R_2 = R_9*R_2; R_7 = 6.0*R_7; R_20 = 17.0*R_20;\n";
                os << "    R_8 = R_20+R_8; R_8 = R_1*R_8; R_1 = R_5*R_12; R_20 = -1.75*R_1;\n";
                os << "    R_20 = R_17+R_20; R_20 = R_10*R_20; R_20 = R_13+R_20; R_13 = R_5*R_20;\n";
                os << "    R_13 = R_14+R_13; R_13 = R_10*R_13; R_13 = 0.25*R_13; R_13 = R_2+R_13;\n";
                os << "    R_13 = 8.0*R_13; R_21 = R_13+R_21; R_20 = 6.0*R_20; R_13 = (-126.0)*R_1;\n";
                os << "    R_18 = R_13+R_18; R_18 = R_10*R_18; R_18 = 0.25*R_18; R_20 = R_18+R_20;\n";
                os << "    R_20 = R_5*R_20; R_13 = R_19+R_13; R_13 = R_9*R_13; R_13 = 0.5*R_13;\n";
                os << "    R_8 = R_13+R_8; R_7 = R_8+R_7; R_7 = R_9*R_7; R_7 = 3.0*R_7;\n";
                os << "    R_20 = R_7+R_20; R_20 = 8.0*R_20; R_3 = R_20+R_3; R_3 = R_5*R_3;\n";
                os << "    R_1 = (-20160.0)*R_1; R_16 = R_1+R_16; R_5 = pow(R_5,3); R_16 = R_5*R_16;\n";
                os << "    R_5 = " << coeff_val << "; R_15 = R_5*R_15; R_15 = (1.0/24.0)*R_15; R_11 = R_5*R_11;\n";
                os << "    R_12 = R_11*R_12; R_12 = (-1.0/512.0)*R_12; R_15 = R_12+R_15; R_21 = R_5*R_21;\n";
                os << "    R_21 = (1.0/1920.0)*R_21; R_15 = R_21+R_15; R_3 = R_5*R_3; R_3 = (1.0/322560.0)*R_3;\n";
                os << "    R_15 = R_3+R_15; R_16 = R_5*R_16; R_16 = (1.0/92897280.0)*R_16; R_15 = R_16+R_15;\n";
                os << "    result = R_15; \n";
                os << "    return result;\n";
                break;
            case 5:
                os << "    TF result; \n";
                os << "    TF R_0 = P1.x; TF R_1 = (-1.0)*R_0; TF R_2 = P0.x; R_1 = R_2+R_1;\n";
                os << "    TF R_3 = R_2+R_0; TF R_4 = pow(R_3,2); R_2 = (-1.0)*R_2; R_0 = R_2+R_0;\n";
                os << "    R_2 = R_0*R_3; TF R_5 = pow(R_0,2); TF R_6 = P0.y; TF R_7 = (-1.0)*R_6;\n";
                os << "    TF R_8 = P1.y; TF R_9 = R_8+R_6; TF R_10 = pow(R_9,2); R_4 = R_10+R_4;\n";
                os << "    R_10 = pow(R_4,3); TF R_11 = pow(R_4,5); TF R_12 = R_1*R_9; R_12 = (-1.0)*R_12;\n";
                os << "    TF R_13 = (-1.0)*R_8; R_6 = R_13+R_6; R_0 = R_0*R_6; R_0 = (-1.0)*R_0;\n";
                os << "    R_3 = R_6*R_3; R_12 = R_3+R_12; R_7 = R_8+R_7; R_9 = R_7*R_9;\n";
                os << "    R_9 = R_2+R_9; R_2 = pow(R_9,2); R_8 = 3.75*R_2; R_3 = 12.0*R_2;\n";
                os << "    R_6 = 48.0*R_2; R_13 = 336.0*R_2; TF R_14 = 50400.0*R_2; TF R_15 = 350.0*R_2;\n";
                os << "    TF R_16 = 7200.0*R_2; TF R_17 = 2.0*R_2; TF R_18 = 2.5*R_2; TF R_19 = 35.0*R_2;\n";
                os << "    R_1 = R_7*R_1; R_1 = R_0+R_1; R_0 = R_1*R_9; TF R_20 = R_4*R_0;\n";
                os << "    TF R_21 = (5.0/128.0)*R_20; R_20 = 0.125*R_20; TF R_22 = 44640.0*R_0; TF R_23 = 92736.0*R_0;\n";
                os << "    TF R_24 = 10740.0*R_0; TF R_25 = 6.0*R_0; TF R_26 = 210.0*R_0; TF R_27 = 864.0*R_0;\n";
                os << "    TF R_28 = 18144000.0*R_0; R_7 = pow(R_7,2); R_5 = R_7+R_5; R_7 = R_5*R_9;\n";
                os << "    R_7 = R_12*R_7; TF R_29 = (-360.0)*R_7; R_7 = -4.5*R_7; TF R_30 = R_5*R_4;\n";
                os << "    TF R_31 = 1.875*R_30; R_31 = R_8+R_31; R_31 = R_0*R_31; R_8 = 6.0*R_30;\n";
                os << "    R_8 = R_3+R_8; R_8 = R_30*R_8; R_3 = 60.0*R_30; R_3 = R_6+R_3;\n";
                os << "    R_3 = R_2*R_3; R_8 = R_3+R_8; R_8 = 0.5*R_8; R_3 = 168.0*R_30;\n";
                os << "    R_3 = R_13+R_3; R_3 = R_30*R_3; R_3 = 0.25*R_3; R_8 = R_3+R_8;\n";
                os << "    R_8 = R_0*R_8; R_8 = 5.0*R_8; R_0 = 25200.0*R_30; R_0 = R_14+R_0;\n";
                os << "    R_0 = R_1*R_0; R_0 = R_9*R_0; R_14 = 760.0*R_30; R_15 = R_14+R_15;\n";
                os << "    R_15 = R_1*R_15; R_14 = 7704.0*R_30; R_14 = R_16+R_14; R_14 = R_1*R_14;\n";
                os << "    R_16 = 0.25*R_30; R_16 = R_17+R_16; R_17 = R_1*R_16; R_17 = R_7+R_17;\n";
                os << "    R_17 = 576.0*R_17; R_16 = R_12*R_16; R_3 = (-5.0/64.0)*R_16; R_3 = R_21+R_3;\n";
                os << "    R_3 = R_10*R_3; R_10 = -0.5*R_16; R_20 = R_10+R_20; R_20 = R_9*R_20;\n";
                os << "    R_10 = 12.0*R_20; R_20 = 3.0*R_20; R_21 = (-288.0)*R_16; R_16 = (-3.0)*R_16;\n";
                os << "    R_13 = 0.5*R_30; R_18 = R_13+R_18; R_18 = R_1*R_18; R_13 = 80.0*R_18;\n";
                os << "    R_13 = R_29+R_13; R_18 = R_7+R_18; R_7 = R_4*R_18; R_7 = 0.25*R_7;\n";
                os << "    R_20 = R_7+R_20; R_20 = R_9*R_20; R_7 = 5.0*R_20; R_20 = 4.0*R_20;\n";
                os << "    R_29 = R_9*R_18; R_29 = 4.0*R_29; R_30 = 22.0*R_30; R_19 = R_30+R_19;\n";
                os << "    R_19 = R_1*R_19; R_1 = pow(R_5,2); R_30 = R_5*R_12; R_2 = (-8208.0)*R_30;\n";
                os << "    R_2 = R_22+R_2; R_2 = R_4*R_2; R_2 = 0.25*R_2; R_2 = R_21+R_2;\n";
                os << "    R_2 = R_5*R_2; R_21 = (-56160.0)*R_30; R_23 = R_21+R_23; R_23 = R_9*R_23;\n";
                os << "    R_23 = 0.5*R_23; R_21 = (-4320.0)*R_30; R_21 = R_24+R_21; R_21 = R_9*R_21;\n";
                os << "    R_21 = 0.5*R_21; R_24 = -2.25*R_30; R_25 = R_24+R_25; R_25 = R_4*R_25;\n";
                os << "    R_25 = R_16+R_25; R_16 = 10.0*R_25; R_25 = R_5*R_25; R_25 = R_29+R_25;\n";
                os << "    R_29 = 0.625*R_25; R_31 = R_29+R_31; R_31 = R_4*R_31; R_31 = R_7+R_31;\n";
                os << "    R_31 = R_4*R_31; R_7 = R_9*R_25; R_7 = 3.0*R_7; R_29 = 0.5*R_25;\n";
                os << "    R_25 = 4.0*R_25; R_24 = (-216.0)*R_30; R_26 = R_24+R_26; R_26 = R_9*R_26;\n";
                os << "    R_26 = 0.5*R_26; R_19 = R_26+R_19; R_26 = 0.25*R_19; R_26 = R_18+R_26;\n";
                os << "    R_26 = R_4*R_26; R_26 = R_10+R_26; R_26 = R_5*R_26; R_7 = R_26+R_7;\n";
                os << "    R_7 = R_9*R_7; R_7 = 2.0*R_7; R_26 = 8.0*R_19; R_13 = R_13+R_26;\n";
                os << "    R_13 = R_9*R_13; R_13 = 0.5*R_13; R_21 = R_26+R_21; R_15 = R_21+R_15;\n";
                os << "    R_15 = 2.0*R_15; R_15 = R_17+R_15; R_15 = R_14+R_15; R_23 = R_15+R_23;\n";
                os << "    R_23 = R_9*R_23; R_23 = 0.5*R_23; R_19 = R_9*R_19; R_27 = R_24+R_27;\n";
                os << "    R_24 = R_4*R_27; R_24 = 0.25*R_24; R_24 = R_16+R_24; R_24 = R_5*R_24;\n";
                os << "    R_13 = R_24+R_13; R_13 = 0.25*R_13; R_13 = R_29+R_13; R_13 = R_4*R_13;\n";
                os << "    R_13 = R_20+R_13; R_13 = R_5*R_13; R_7 = R_13+R_7; R_7 = 10.0*R_7;\n";
                os << "    R_8 = R_7+R_8; R_27 = R_5*R_27; R_27 = R_4*R_27; R_27 = 0.25*R_27;\n";
                os << "    R_19 = R_27+R_19; R_25 = R_19+R_25; R_25 = 6.0*R_25; R_23 = R_25+R_23;\n";
                os << "    R_2 = R_23+R_2; R_2 = 10.0*R_2; R_0 = R_2+R_0; R_0 = R_1*R_0;\n";
                os << "    R_30 = (-1814400.0)*R_30; R_28 = R_30+R_28; R_5 = pow(R_5,4); R_28 = R_5*R_28;\n";
                os << "    R_5 = " << coeff_val << "; R_3 = R_5*R_3; R_3 = (1.0/24.0)*R_3; R_11 = R_5*R_11;\n";
                os << "    R_12 = R_11*R_12; R_12 = (-1.0/2048.0)*R_12; R_3 = R_12+R_3; R_31 = R_5*R_31;\n";
                os << "    R_31 = (1.0/1920.0)*R_31; R_3 = R_31+R_3; R_8 = R_5*R_8; R_8 = (1.0/322560.0)*R_8;\n";
                os << "    R_3 = R_8+R_3; R_0 = R_5*R_0; R_0 = (1.0/92897280.0)*R_0; R_3 = R_0+R_3;\n";
                os << "    R_28 = R_5*R_28; R_28 = (1.0/40874803200.0)*R_28; R_3 = R_28+R_3; result = R_3;\n";
                os << "    return result;\n";
                break;
            case 6:
                os << "    TF result; \n";
                os << "    TF R_0 = P1.x; TF R_1 = (-1.0)*R_0; TF R_2 = P0.x; R_1 = R_2+R_1;\n";
                os << "    TF R_3 = R_2+R_0; TF R_4 = pow(R_3,2); R_2 = (-1.0)*R_2; R_0 = R_2+R_0;\n";
                os << "    R_2 = R_0*R_3; TF R_5 = pow(R_0,2); TF R_6 = P0.y; TF R_7 = (-1.0)*R_6;\n";
                os << "    TF R_8 = P1.y; TF R_9 = R_8+R_6; TF R_10 = pow(R_9,2); R_4 = R_10+R_4;\n";
                os << "    R_10 = pow(R_4,4); TF R_11 = pow(R_4,6); TF R_12 = pow(R_4,2); TF R_13 = R_1*R_9;\n";
                os << "    R_13 = (-1.0)*R_13; TF R_14 = (-1.0)*R_8; R_6 = R_14+R_6; R_0 = R_6*R_0;\n";
                os << "    R_0 = (-1.0)*R_0; R_3 = R_6*R_3; R_13 = R_3+R_13; R_7 = R_8+R_7;\n";
                os << "    R_9 = R_7*R_9; R_2 = R_9+R_2; R_9 = R_4*R_2; R_8 = pow(R_2,2);\n";
                os << "    R_3 = 1.875*R_8; R_6 = 200.0*R_8; R_14 = 120.0*R_8; TF R_15 = 26640.0*R_8;\n";
                os << "    TF R_16 = 30240.0*R_8; TF R_17 = 26592.0*R_8; TF R_18 = 7344.0*R_8; TF R_19 = 34560.0*R_8;\n";
                os << "    TF R_20 = 54.0*R_8; TF R_21 = 2340.0*R_8; TF R_22 = 3.0*R_8; TF R_23 = 7257600.0*R_8;\n";
                os << "    TF R_24 = 458064.0*R_8; TF R_25 = 432.0*R_8; TF R_26 = 2.5*R_8; R_1 = R_1*R_7;\n";
                os << "    R_1 = R_0+R_1; R_0 = R_1*R_4; R_0 = R_2*R_0; R_0 = (3.0/256.0)*R_0;\n";
                os << "    R_9 = R_1*R_9; R_9 = 0.125*R_9; TF R_27 = R_1*R_2; TF R_28 = R_4*R_27;\n";
                os << "    TF R_29 = 82920.0*R_27; TF R_30 = 272016.0*R_27; TF R_31 = 7.25*R_27; TF R_32 = 1254.0*R_27;\n";
                os << "    TF R_33 = 81144.0*R_27; TF R_34 = 72000.0*R_27; TF R_35 = 50976.0*R_27; TF R_36 = 5328.0*R_27;\n";
                os << "    TF R_37 = 6000.0*R_27; TF R_38 = 300.0*R_27; TF R_39 = 3801600.0*R_27; TF R_40 = 972000.0*R_27;\n";
                os << "    TF R_41 = 9879792.0*R_27; TF R_42 = 7416.0*R_27; R_27 = 2874009600.0*R_27; R_7 = pow(R_7,2);\n";
                os << "    R_5 = R_7+R_5; R_7 = R_5*R_4; TF R_43 = (45.0/64.0)*R_7; R_43 = R_3+R_43;\n";
                os << "    R_43 = R_1*R_43; R_3 = 75.0*R_7; R_3 = R_6+R_3; R_3 = R_4*R_3;\n";
                os << "    R_3 = R_5*R_3; R_6 = 100.0*R_7; R_6 = R_14+R_6; R_6 = R_8*R_6;\n";
                os << "    R_3 = R_6+R_3; R_28 = R_3*R_28; R_28 = 1.5*R_28; R_3 = 18900.0*R_7;\n";
                os << "    R_3 = R_15+R_3; R_3 = R_4*R_3; R_3 = R_5*R_3; R_15 = 48960.0*R_7;\n";
                os << "    R_15 = R_16+R_15; R_15 = R_8*R_15; R_3 = R_15+R_3; R_3 = R_1*R_3;\n";
                os << "    R_3 = R_2*R_3; R_15 = 19680.0*R_7; R_17 = R_15+R_17; R_17 = R_1*R_17;\n";
                os << "    R_15 = 5760.0*R_7; R_15 = R_18+R_15; R_15 = R_1*R_15; R_18 = 10080.0*R_7;\n";
                os << "    R_19 = R_18+R_19; R_19 = R_1*R_19; R_18 = 27.0*R_7; R_20 = R_18+R_20;\n";
                os << "    R_20 = R_1*R_20; R_18 = 750.0*R_7; R_18 = R_21+R_18; R_18 = R_1*R_18;\n";
                os << "    R_21 = 0.5*R_7; R_22 = R_21+R_22; R_22 = R_1*R_22; R_21 = 140.0*R_22;\n";
                os << "    R_8 = 2721600.0*R_7; R_23 = R_8+R_23; R_23 = R_2*R_23; R_23 = R_1*R_23;\n";
                os << "    R_8 = 594336.0*R_7; R_24 = R_8+R_24; R_24 = R_1*R_24; R_8 = 1008.0*R_7;\n";
                os << "    R_25 = R_8+R_25; R_25 = R_1*R_25; R_8 = 0.25*R_7; R_8 = R_26+R_8;\n";
                os << "    R_26 = R_13*R_8; R_16 = (-3.0/128.0)*R_26; R_0 = R_16+R_0; R_0 = R_10*R_0;\n";
                os << "    R_10 = (-768.0)*R_26; R_16 = (-4.0)*R_26; R_6 = (-2880.0)*R_26; R_14 = (-216.0)*R_26;\n";
                os << "    TF R_44 = -0.5*R_26; R_44 = R_9+R_44; R_44 = R_2*R_44; R_9 = 9.0*R_44;\n";
                os << "    TF R_45 = 24.0*R_44; TF R_46 = 1152.0*R_44; R_44 = 4.0*R_44; R_26 = (-10080.0)*R_26;\n";
                os << "    R_8 = R_1*R_8; R_1 = R_5*R_2; R_1 = R_13*R_1; TF R_47 = (-770.0)*R_1;\n";
                os << "    R_21 = R_47+R_21; R_1 = -5.5*R_1; R_22 = R_1+R_22; R_47 = (9.0/16.0)*R_22;\n";
                os << "    R_47 = R_43+R_47; R_47 = R_4*R_47; R_47 = R_9+R_47; R_47 = R_2*R_47;\n";
                os << "    R_9 = 1.5*R_22; R_43 = R_2*R_22; R_43 = 5.0*R_43; TF R_48 = 120.0*R_22;\n";
                os << "    R_22 = R_4*R_22; R_22 = 0.25*R_22; R_44 = R_22+R_44; R_44 = R_2*R_44;\n";
                os << "    R_22 = 12.0*R_44; TF R_49 = 3.0*R_44; R_44 = 36.0*R_44; R_8 = R_1+R_8;\n";
                os << "    R_8 = 1536.0*R_8; R_1 = pow(R_5,3); TF R_50 = R_5*R_13; TF R_51 = (-16632.0)*R_50;\n";
                os << "    R_29 = R_51+R_29; R_29 = R_4*R_29; R_29 = 0.25*R_29; R_29 = R_10+R_29;\n";
                os << "    R_29 = R_5*R_29; R_10 = (-192720.0)*R_50; R_30 = R_10+R_30; R_30 = R_2*R_30;\n";
                os << "    R_30 = 0.5*R_30; R_30 = R_17+R_30; R_30 = R_2*R_30; R_30 = 0.5*R_30;\n";
                os << "    R_17 = -2.75*R_50; R_17 = R_31+R_17; R_17 = R_4*R_17; R_17 = R_16+R_17;\n";
                os << "    R_16 = 14.0*R_17; R_17 = R_5*R_17; R_17 = R_43+R_17; R_43 = R_4*R_17;\n";
                os << "    R_31 = (3.0/16.0)*R_43; R_47 = R_31+R_47; R_47 = R_12*R_47; R_43 = 0.25*R_43;\n";
                os << "    R_49 = R_43+R_49; R_49 = R_2*R_49; R_49 = 2.0*R_49; R_43 = R_2*R_17;\n";
                os << "    R_43 = 4.0*R_43; R_12 = 6.0*R_17; R_31 = (-54648.0)*R_50; R_31 = R_33+R_31;\n";
                os << "    R_31 = R_2*R_31; R_31 = 0.5*R_31; R_15 = R_31+R_15; R_15 = R_4*R_15;\n";
                os << "    R_15 = 0.25*R_15; R_15 = R_46+R_15; R_15 = R_5*R_15; R_46 = (-158400.0)*R_50;\n";
                os << "    R_46 = R_34+R_46; R_46 = R_2*R_46; R_46 = 0.5*R_46; R_46 = R_19+R_46;\n";
                os << "    R_46 = R_2*R_46; R_46 = 0.5*R_46; R_19 = (-15840.0)*R_50; R_19 = R_35+R_19;\n";
                os << "    R_19 = R_4*R_19; R_19 = 0.25*R_19; R_19 = R_6+R_19; R_19 = R_5*R_19;\n";
                os << "    R_46 = R_19+R_46; R_46 = R_2*R_46; R_46 = 0.5*R_46; R_15 = R_46+R_15;\n";
                os << "    R_15 = R_2*R_15; R_15 = 0.5*R_15; R_46 = (-1584.0)*R_50; R_36 = R_46+R_36;\n";
                os << "    R_36 = R_4*R_36; R_36 = 0.25*R_36; R_36 = R_14+R_36; R_36 = R_5*R_36;\n";
                os << "    R_14 = (-11220.0)*R_50; R_14 = R_37+R_14; R_14 = R_2*R_14; R_14 = 0.5*R_14;\n";
                os << "    R_37 = (-330.0)*R_50; R_32 = R_37+R_32; R_46 = R_4*R_32; R_46 = 0.25*R_46;\n";
                os << "    R_46 = R_16+R_46; R_46 = R_5*R_46; R_32 = R_5*R_32; R_32 = R_4*R_32;\n";
                os << "    R_32 = 0.25*R_32; R_38 = R_37+R_38; R_38 = R_2*R_38; R_38 = 0.5*R_38;\n";
                os << "    R_20 = R_38+R_20; R_38 = 0.25*R_20; R_38 = R_9+R_38; R_38 = R_4*R_38;\n";
                os << "    R_38 = R_45+R_38; R_38 = R_5*R_38; R_43 = R_38+R_43; R_38 = R_2*R_43;\n";
                os << "    R_38 = 3.0*R_38; R_43 = R_4*R_43; R_43 = 0.25*R_43; R_49 = R_43+R_49;\n";
                os << "    R_49 = R_2*R_49; R_43 = R_2*R_20; R_32 = R_43+R_32; R_12 = R_32+R_12;\n";
                os << "    R_12 = 10.0*R_12; R_30 = R_12+R_30; R_29 = R_30+R_29; R_29 = R_7*R_29;\n";
                os << "    R_29 = 0.25*R_29; R_20 = 10.0*R_20; R_21 = R_21+R_20; R_21 = R_2*R_21;\n";
                os << "    R_21 = 0.5*R_21; R_21 = R_46+R_21; R_21 = 0.25*R_21; R_21 = R_17+R_21;\n";
                os << "    R_21 = R_4*R_21; R_21 = R_22+R_21; R_21 = R_5*R_21; R_38 = R_21+R_38;\n";
                os << "    R_38 = R_4*R_38; R_38 = 0.25*R_38; R_38 = R_49+R_38; R_38 = 12.0*R_38;\n";
                os << "    R_28 = R_38+R_28; R_48 = R_20+R_48; R_48 = R_18+R_48; R_14 = R_48+R_14;\n";
                os << "    R_14 = R_2*R_14; R_14 = 0.5*R_14; R_36 = R_14+R_36; R_36 = R_4*R_36;\n";
                os << "    R_36 = 0.25*R_36; R_36 = R_44+R_36; R_36 = R_5*R_36; R_15 = R_36+R_15;\n";
                os << "    R_15 = 2.0*R_15; R_29 = R_15+R_29; R_29 = 12.0*R_29; R_3 = R_29+R_3;\n";
                os << "    R_3 = R_5*R_3; R_29 = (-475200.0)*R_50; R_29 = R_39+R_29; R_29 = R_7*R_29;\n";
                os << "    R_29 = 0.25*R_29; R_7 = (-205920.0)*R_50; R_7 = R_40+R_7; R_7 = R_4*R_7;\n";
                os << "    R_7 = 0.25*R_7; R_7 = R_26+R_7; R_7 = R_5*R_7; R_26 = (-4378704.0)*R_50;\n";
                os << "    R_26 = R_41+R_26; R_26 = R_2*R_26; R_26 = 0.5*R_26; R_41 = (-3960.0)*R_50;\n";
                os << "    R_41 = R_42+R_41; R_41 = R_2*R_41; R_41 = 0.5*R_41; R_25 = R_41+R_25;\n";
                os << "    R_25 = 10.0*R_25; R_25 = R_8+R_25; R_26 = R_25+R_26; R_24 = R_26+R_24;\n";
                os << "    R_24 = R_2*R_24; R_24 = 0.5*R_24; R_7 = R_24+R_7; R_7 = 2.0*R_7;\n";
                os << "    R_29 = R_7+R_29; R_29 = 12.0*R_29; R_23 = R_29+R_23; R_23 = R_1*R_23;\n";
                os << "    R_50 = (-239500800.0)*R_50; R_27 = R_50+R_27; R_5 = pow(R_5,5); R_27 = R_5*R_27;\n";
                os << "    R_5 = " << coeff_val << "; R_0 = R_5*R_0; R_0 = (1.0/24.0)*R_0; R_11 = R_5*R_11;\n";
                os << "    R_13 = R_11*R_13; R_13 = (-1.0/8192.0)*R_13; R_0 = R_13+R_0; R_47 = R_5*R_47;\n";
                os << "    R_47 = (1.0/1920.0)*R_47; R_0 = R_47+R_0; R_28 = R_5*R_28; R_28 = (1.0/322560.0)*R_28;\n";
                os << "    R_0 = R_28+R_0; R_3 = R_5*R_3; R_3 = (1.0/92897280.0)*R_3; R_0 = R_3+R_0;\n";
                os << "    R_23 = R_5*R_23; R_23 = (1.0/40874803200.0)*R_23; R_0 = R_23+R_0; R_27 = R_5*R_27;\n";
                os << "    R_27 = (1.0/25505877196800.0)*R_27; R_0 = R_27+R_0; result = R_0; \n";
                os << "    return result;\n";
                break;
            case 7:
                os << "    TF result; \n";
                os << "    TF R_0 = P0.y; TF R_1 = (-1.0)*R_0; TF R_2 = P1.y; TF R_3 = R_2+R_0;\n";
                os << "    TF R_4 = pow(R_3,2); TF R_5 = (-1.0)*R_2; R_5 = R_0+R_5; R_1 = R_2+R_1;\n";
                os << "    R_2 = R_1*R_3; R_0 = pow(R_1,2); TF R_6 = P0.x; TF R_7 = (-1.0)*R_6;\n";
                os << "    TF R_8 = P1.x; TF R_9 = (-1.0)*R_8; R_9 = R_6+R_9; R_1 = R_9*R_1;\n";
                os << "    R_3 = R_9*R_3; R_3 = (-1.0)*R_3; R_6 = R_8+R_6; R_9 = pow(R_6,2);\n";
                os << "    R_9 = R_4+R_9; R_4 = pow(R_9,5); TF R_10 = pow(R_9,7); TF R_11 = pow(R_9,3);\n";
                os << "    TF R_12 = R_5*R_6; R_3 = R_12+R_3; R_7 = R_8+R_7; R_6 = R_7*R_6;\n";
                os << "    R_6 = R_2+R_6; R_2 = pow(R_6,2); R_8 = (105.0/128.0)*R_2; R_12 = 1320.0*R_2;\n";
                os << "    TF R_13 = 30.0*R_2; TF R_14 = 4140.0*R_2; TF R_15 = 240.0*R_2; TF R_16 = 46680.0*R_2;\n";
                os << "    TF R_17 = 77760.0*R_2; TF R_18 = 1417.5*R_2; TF R_19 = 36120.0*R_2; TF R_20 = 2800.75*R_2;\n";
                os << "    TF R_21 = 1386.0*R_2; TF R_22 = 2270880.0*R_2; TF R_23 = 4354560.0*R_2; TF R_24 = 3259248.0*R_2;\n";
                os << "    TF R_25 = 1208720.0*R_2; TF R_26 = 170680.0*R_2; TF R_27 = 6771456.0*R_2; TF R_28 = 89624.0*R_2;\n";
                os << "    TF R_29 = 22680.0*R_2; TF R_30 = 995952.0*R_2; TF R_31 = 7560.0*R_2; TF R_32 = 120634560.0*R_2;\n";
                os << "    TF R_33 = 2310.0*R_2; TF R_34 = 33880.0*R_2; TF R_35 = 77.0*R_2; TF R_36 = 3.5*R_2;\n";
                os << "    TF R_37 = 1680.0*R_2; TF R_38 = 3.0*R_2; TF R_39 = 1397088000.0*R_2; R_5 = R_5*R_7;\n";
                os << "    R_5 = (-1.0)*R_5; R_5 = R_1+R_5; R_1 = R_5*R_9; R_1 = R_6*R_1;\n";
                os << "    TF R_40 = (7.0/2048.0)*R_1; R_1 = 0.125*R_1; TF R_41 = R_5*R_6; TF R_42 = 6149.5*R_41;\n";
                os << "    TF R_43 = 7180.0*R_41; TF R_44 = 9578.75*R_41; TF R_45 = 3073.5*R_41; TF R_46 = 28040.0*R_41;\n";
                os << "    TF R_47 = 31332.0*R_41; TF R_48 = 53255904.0*R_41; TF R_49 = 14514912.0*R_41; TF R_50 = 6715040.0*R_41;\n";
                os << "    TF R_51 = 2336160.0*R_41; TF R_52 = 3861328.0*R_41; TF R_53 = 15467328.0*R_41; TF R_54 = 613040.0*R_41;\n";
                os << "    TF R_55 = 48720.0*R_41; TF R_56 = 34725504.0*R_41; TF R_57 = 23191968.0*R_41; TF R_58 = 83776.0*R_41;\n";
                os << "    TF R_59 = 914094720.0*R_41; TF R_60 = 4133079936.0*R_41; TF R_61 = 134472.0*R_41; TF R_62 = 418880.0*R_41;\n";
                os << "    TF R_63 = 36204.0*R_41; TF R_64 = 406.0*R_41; TF R_65 = 4872.0*R_41; TF R_66 = 1716.0*R_41;\n";
                os << "    TF R_67 = 8.5*R_41; TF R_68 = 610248038400.0*R_41; R_7 = pow(R_7,2); R_0 = R_7+R_0;\n";
                os << "    R_7 = pow(R_0,2); TF R_69 = R_0*R_6; R_69 = R_3*R_69; TF R_70 = (-1404.0)*R_69;\n";
                os << "    R_69 = -6.5*R_69; TF R_71 = R_0*R_9; TF R_72 = (63.0/256.0)*R_71; R_72 = R_8+R_72;\n";
                os << "    R_72 = R_5*R_72; R_8 = 396.0*R_71; R_8 = R_12+R_8; R_8 = R_71*R_8;\n";
                os << "    R_12 = (7.0/64.0)*R_8; R_8 = 0.25*R_8; TF R_73 = 9.0*R_71; R_73 = R_13+R_73;\n";
                os << "    R_73 = R_71*R_73; R_13 = 2100.0*R_71; R_13 = R_14+R_13; R_13 = R_9*R_13;\n";
                os << "    R_13 = R_0*R_13; R_14 = 150.0*R_71; R_14 = R_15+R_14; R_14 = R_2*R_14;\n";
                os << "    R_73 = R_14+R_73; R_15 = (21.0/32.0)*R_73; R_12 = R_15+R_12; R_12 = R_41*R_12;\n";
                os << "    R_73 = 1.5*R_73; R_8 = R_73+R_8; R_8 = R_71*R_8; R_14 = 6.0*R_14;\n";
                os << "    R_13 = R_14+R_13; R_13 = R_2*R_13; R_8 = R_13+R_8; R_8 = 0.5*R_8;\n";
                os << "    R_13 = 24300.0*R_71; R_13 = R_16+R_13; R_13 = R_71*R_13; R_16 = 82920.0*R_71;\n";
                os << "    R_17 = R_16+R_17; R_17 = R_2*R_17; R_13 = R_17+R_13; R_13 = R_71*R_13;\n";
                os << "    R_13 = (1.0/16.0)*R_13; R_8 = R_13+R_8; R_8 = R_41*R_8; R_8 = 7.0*R_8;\n";
                os << "    R_41 = 801.0*R_71; R_18 = R_41+R_18; R_18 = R_5*R_18; R_41 = 7680.0*R_71;\n";
                os << "    R_19 = R_41+R_19; R_19 = R_5*R_19; R_41 = 1336.5*R_71; R_20 = R_41+R_20;\n";
                os << "    R_20 = R_5*R_20; R_41 = 1668.0*R_71; R_21 = R_41+R_21; R_21 = R_5*R_21;\n";
                os << "    R_41 = 1360800.0*R_71; R_22 = R_41+R_22; R_22 = R_71*R_22; R_41 = 4986720.0*R_71;\n";
                os << "    R_23 = R_41+R_23; R_23 = R_2*R_23; R_22 = R_23+R_22; R_22 = R_5*R_22;\n";
                os << "    R_22 = R_6*R_22; R_22 = 1.75*R_22; R_23 = 2924064.0*R_71; R_24 = R_23+R_24;\n";
                os << "    R_24 = R_5*R_24; R_23 = 502560.0*R_71; R_25 = R_23+R_25; R_25 = R_5*R_25;\n";
                os << "    R_23 = 200736.0*R_71; R_26 = R_23+R_26; R_26 = R_5*R_26; R_23 = 2681280.0*R_71;\n";
                os << "    R_27 = R_23+R_27; R_27 = R_5*R_27; R_23 = 42768.0*R_71; R_28 = R_23+R_28;\n";
                os << "    R_28 = R_5*R_28; R_23 = 5760.0*R_71; R_29 = R_23+R_29; R_29 = R_5*R_29;\n";
                os << "    R_23 = 1202112.0*R_71; R_23 = R_30+R_23; R_23 = R_5*R_23; R_30 = 5112.0*R_71;\n";
                os << "    R_30 = R_31+R_30; R_30 = R_5*R_30; R_31 = 198709632.0*R_71; R_32 = R_31+R_32;\n";
                os << "    R_32 = R_5*R_32; R_31 = 2052.0*R_71; R_33 = R_31+R_33; R_33 = R_5*R_33;\n";
                os << "    R_31 = 25000.0*R_71; R_34 = R_31+R_34; R_34 = R_5*R_34; R_31 = 32.0*R_71;\n";
                os << "    R_35 = R_31+R_35; R_35 = R_5*R_35; R_31 = 0.5*R_71; R_36 = R_31+R_36;\n";
                os << "    R_36 = R_5*R_36; R_31 = 216.0*R_36; R_31 = R_70+R_31; R_36 = R_69+R_36;\n";
                os << "    R_69 = (7.0/32.0)*R_36; R_69 = R_72+R_69; R_69 = R_9*R_69; R_72 = 120.0*R_36;\n";
                os << "    R_70 = 92160.0*R_36; R_2 = 6144.0*R_36; R_41 = R_9*R_36; R_41 = 0.25*R_41;\n";
                os << "    R_13 = R_6*R_36; R_13 = 6.0*R_13; R_17 = 2.0*R_36; R_36 = 3024.0*R_36;\n";
                os << "    R_16 = 492.0*R_71; R_37 = R_16+R_37; R_37 = R_5*R_37; R_16 = 0.25*R_71;\n";
                os << "    R_16 = R_38+R_16; R_16 = R_3*R_16; R_38 = (-7.0/1024.0)*R_16; R_40 = R_38+R_40;\n";
                os << "    R_40 = R_4*R_40; R_4 = (-1700.0)*R_16; R_38 = (-225.0)*R_16; R_14 = (-1600.0)*R_16;\n";
                os << "    R_73 = (-97920.0)*R_16; R_15 = (-60000.0)*R_16; TF R_74 = (-452160.0)*R_16; TF R_75 = -0.5*R_16;\n";
                os << "    R_75 = R_1+R_75; R_75 = R_6*R_75; R_1 = 4.375*R_75; R_69 = R_1+R_69;\n";
                os << "    R_69 = R_6*R_69; R_1 = 1280.0*R_75; TF R_76 = 13440.0*R_75; TF R_77 = 115200.0*R_75;\n";
                os << "    TF R_78 = 5.0*R_75; R_78 = R_41+R_78; R_41 = R_0*R_78; R_78 = R_6*R_78;\n";
                os << "    TF R_79 = 72.0*R_78; TF R_80 = 4.0*R_78; TF R_81 = 2304.0*R_78; R_78 = 24.0*R_78;\n";
                os << "    R_75 = 40.0*R_75; TF R_82 = (-1762560.0)*R_16; R_16 = (-5.0)*R_16; R_71 = 419126400.0*R_71;\n";
                os << "    R_71 = R_39+R_71; R_71 = R_6*R_71; R_71 = R_5*R_71; R_5 = pow(R_0,4);\n";
                os << "    R_39 = R_0*R_3; TF R_83 = (-4862.0)*R_39; R_83 = R_42+R_83; R_83 = R_6*R_83;\n";
                os << "    R_42 = (-2275.0)*R_39; R_43 = R_42+R_43; R_43 = R_9*R_43; R_43 = R_4+R_43;\n";
                os << "    R_43 = R_0*R_43; R_4 = (-156000.0)*R_39; R_4 = R_55+R_4; R_4 = R_6*R_4;\n";
                os << "    R_4 = 0.5*R_4; R_42 = (-8775.0)*R_39; R_44 = R_42+R_44; R_44 = R_6*R_44;\n";
                os << "    R_44 = R_20+R_44; R_44 = R_6*R_44; R_20 = -731.25*R_39; R_20 = R_45+R_20;\n";
                os << "    R_20 = R_9*R_20; R_38 = R_20+R_38; R_38 = R_0*R_38; R_44 = R_38+R_44;\n";
                os << "    R_44 = R_9*R_44; R_44 = R_79+R_44; R_44 = R_0*R_44; R_79 = (-8840.0)*R_39;\n";
                os << "    R_46 = R_79+R_46; R_46 = R_9*R_46; R_46 = 0.25*R_46; R_46 = R_14+R_46;\n";
                os << "    R_46 = R_0*R_46; R_14 = (-14976.0)*R_39; R_14 = R_47+R_14; R_14 = R_6*R_14;\n";
                os << "    R_14 = 0.5*R_14; R_47 = (-27855360.0)*R_39; R_47 = R_48+R_47; R_47 = R_6*R_47;\n";
                os << "    R_47 = 0.5*R_47; R_48 = (-2602080.0)*R_39; R_48 = R_49+R_48; R_48 = R_9*R_48;\n";
                os << "    R_48 = 0.25*R_48; R_48 = R_73+R_48; R_48 = R_0*R_48; R_73 = (-7038720.0)*R_39;\n";
                os << "    R_73 = R_50+R_73; R_73 = R_6*R_73; R_73 = 0.5*R_73; R_73 = R_25+R_73;\n";
                os << "    R_73 = R_6*R_73; R_73 = 0.5*R_73; R_25 = (-586560.0)*R_39; R_25 = R_51+R_25;\n";
                os << "    R_25 = R_9*R_25; R_25 = 0.25*R_25; R_25 = R_15+R_25; R_25 = R_0*R_25;\n";
                os << "    R_73 = R_25+R_73; R_73 = R_6*R_73; R_73 = 0.5*R_73; R_25 = (-1716000.0)*R_39;\n";
                os << "    R_25 = R_52+R_25; R_25 = R_6*R_25; R_25 = 0.5*R_25; R_25 = R_26+R_25;\n";
                os << "    R_25 = R_9*R_25; R_25 = 0.25*R_25; R_25 = R_76+R_25; R_25 = R_0*R_25;\n";
                os << "    R_73 = R_25+R_73; R_73 = 2.0*R_73; R_25 = (-3983616.0)*R_39; R_25 = R_53+R_25;\n";
                os << "    R_25 = R_9*R_25; R_25 = 0.25*R_25; R_25 = R_74+R_25; R_25 = R_0*R_25;\n";
                os << "    R_74 = (-561600.0)*R_39; R_54 = R_74+R_54; R_54 = R_6*R_54; R_54 = 0.5*R_54;\n";
                os << "    R_54 = R_28+R_54; R_28 = 6.0*R_54; R_24 = R_28+R_24; R_47 = R_24+R_47;\n";
                os << "    R_47 = R_6*R_47; R_47 = 0.5*R_47; R_47 = R_48+R_47; R_47 = R_9*R_47;\n";
                os << "    R_47 = 0.25*R_47; R_47 = R_81+R_47; R_47 = R_0*R_47; R_54 = 12.0*R_54;\n";
                os << "    R_81 = (-106080.0)*R_39; R_81 = R_55+R_81; R_81 = R_6*R_81; R_81 = 0.5*R_81;\n";
                os << "    R_81 = R_29+R_81; R_29 = 0.25*R_81; R_55 = R_6*R_81; R_55 = 0.5*R_55;\n";
                os << "    R_55 = R_46+R_55; R_55 = R_6*R_55; R_55 = 0.5*R_55; R_81 = 8.0*R_81;\n";
                os << "    R_70 = R_81+R_70; R_54 = R_70+R_54; R_70 = (-38613120.0)*R_39; R_56 = R_70+R_56;\n";
                os << "    R_56 = R_6*R_56; R_56 = 0.5*R_56; R_70 = (-10236096.0)*R_39; R_70 = R_57+R_70;\n";
                os << "    R_70 = R_6*R_70; R_70 = 0.5*R_70; R_57 = (-57200.0)*R_39; R_57 = R_58+R_57;\n";
                os << "    R_57 = R_6*R_57; R_57 = 0.5*R_57; R_30 = R_57+R_30; R_30 = 24.0*R_30;\n";
                os << "    R_57 = (-117037440.0)*R_39; R_59 = R_57+R_59; R_59 = R_9*R_59; R_59 = 0.25*R_59;\n";
                os << "    R_59 = R_82+R_59; R_59 = R_0*R_59; R_82 = (-1404449280.0)*R_39; R_82 = R_60+R_82;\n";
                os << "    R_82 = R_6*R_82; R_82 = 0.5*R_82; R_82 = R_32+R_82; R_82 = R_6*R_82;\n";
                os << "    R_82 = 0.5*R_82; R_32 = (-27144.0)*R_39; R_61 = R_32+R_61; R_61 = R_9*R_61;\n";
                os << "    R_61 = 0.25*R_61; R_32 = (-271440.0)*R_39; R_32 = R_62+R_32; R_32 = R_6*R_32;\n";
                os << "    R_32 = 0.5*R_32; R_62 = (-20592.0)*R_39; R_63 = R_62+R_63; R_63 = R_6*R_63;\n";
                os << "    R_63 = 0.5*R_63; R_33 = R_63+R_33; R_63 = 0.25*R_33; R_18 = R_63+R_18;\n";
                os << "    R_83 = R_18+R_83; R_83 = R_9*R_83; R_83 = R_1+R_83; R_83 = R_0*R_83;\n";
                os << "    R_33 = 2.0*R_33; R_32 = R_33+R_32; R_32 = R_34+R_32; R_34 = (-8424.0)*R_39;\n";
                os << "    R_65 = R_34+R_65; R_65 = R_6*R_65; R_65 = 0.5*R_65; R_37 = R_65+R_37;\n";
                os << "    R_65 = 48.0*R_37; R_65 = R_54+R_65; R_65 = R_56+R_65; R_27 = R_65+R_27;\n";
                os << "    R_27 = R_6*R_27; R_27 = 0.5*R_27; R_25 = R_27+R_25; R_25 = R_6*R_25;\n";
                os << "    R_25 = 0.5*R_25; R_37 = R_6*R_37; R_37 = 0.5*R_37; R_27 = (-468.0)*R_39;\n";
                os << "    R_64 = R_64+R_27; R_64 = R_6*R_64; R_64 = 0.5*R_64; R_35 = R_64+R_35;\n";
                os << "    R_64 = 120.0*R_35; R_19 = R_64+R_19; R_4 = R_19+R_4; R_4 = 0.125*R_4;\n";
                os << "    R_72 = R_4+R_72; R_72 = R_29+R_72; R_72 = R_6*R_72; R_43 = R_72+R_43;\n";
                os << "    R_43 = R_6*R_43; R_83 = R_43+R_83; R_83 = 0.5*R_83; R_43 = 12.0*R_35;\n";
                os << "    R_31 = R_31+R_43; R_31 = R_6*R_31; R_31 = 0.5*R_31; R_14 = R_43+R_14;\n";
                os << "    R_21 = R_14+R_21; R_21 = R_9*R_21; R_21 = 0.25*R_21; R_14 = 768.0*R_35;\n";
                os << "    R_30 = R_14+R_30; R_30 = R_2+R_30; R_23 = R_30+R_23; R_70 = R_23+R_70;\n";
                os << "    R_70 = R_9*R_70; R_70 = 0.25*R_70; R_70 = R_77+R_70; R_70 = R_0*R_70;\n";
                os << "    R_77 = 0.25*R_35; R_77 = R_17+R_77; R_77 = R_9*R_77; R_77 = R_75+R_77;\n";
                os << "    R_75 = 14.0*R_77; R_21 = R_75+R_21; R_21 = R_0*R_21; R_55 = R_21+R_55;\n";
                os << "    R_55 = R_0*R_55; R_55 = R_9*R_55; R_55 = 0.25*R_55; R_77 = R_0*R_77;\n";
                os << "    R_35 = 28.0*R_35; R_32 = R_35+R_32; R_32 = R_36+R_32; R_32 = R_6*R_32;\n";
                os << "    R_32 = 0.5*R_32; R_66 = R_27+R_66; R_66 = R_9*R_66; R_66 = 0.25*R_66;\n";
                os << "    R_27 = -3.25*R_39; R_67 = R_27+R_67; R_67 = R_9*R_67; R_67 = R_16+R_67;\n";
                os << "    R_16 = R_0*R_67; R_16 = R_13+R_16; R_13 = R_9*R_16; R_27 = (7.0/128.0)*R_13;\n";
                os << "    R_69 = R_27+R_69; R_69 = R_11*R_69; R_13 = 0.25*R_13; R_80 = R_13+R_80;\n";
                os << "    R_80 = R_6*R_80; R_80 = 3.0*R_80; R_13 = 1.5*R_16; R_16 = R_6*R_16;\n";
                os << "    R_11 = 0.5*R_16; R_41 = R_11+R_41; R_41 = 4608.0*R_41; R_73 = R_41+R_73;\n";
                os << "    R_73 = R_25+R_73; R_73 = R_70+R_73; R_73 = R_6*R_73; R_73 = 0.5*R_73;\n";
                os << "    R_16 = 5.0*R_16; R_16 = R_77+R_16; R_77 = 10.0*R_16; R_77 = R_83+R_77;\n";
                os << "    R_77 = R_6*R_77; R_77 = R_44+R_77; R_77 = R_9*R_77; R_44 = R_9*R_16;\n";
                os << "    R_44 = 0.25*R_44; R_80 = R_44+R_80; R_44 = R_6*R_80; R_83 = 7.0*R_44;\n";
                os << "    R_44 = 4.0*R_44; R_77 = R_44+R_77; R_77 = R_0*R_77; R_80 = R_0*R_80;\n";
                os << "    R_16 = R_6*R_16; R_16 = 4.0*R_16; R_44 = 252.0*R_67; R_61 = R_44+R_61;\n";
                os << "    R_61 = R_0*R_61; R_67 = 18.0*R_67; R_66 = R_67+R_66; R_66 = R_0*R_66;\n";
                os << "    R_31 = R_66+R_31; R_31 = 0.25*R_31; R_31 = R_13+R_31; R_31 = R_9*R_31;\n";
                os << "    R_31 = R_78+R_31; R_31 = R_0*R_31; R_31 = R_31+R_16; R_67 = 0.875*R_31;\n";
                os << "    R_67 = R_12+R_67; R_67 = R_9*R_67; R_67 = R_83+R_67; R_67 = R_9*R_67;\n";
                os << "    R_31 = R_6*R_31; R_31 = R_55+R_31; R_37 = R_66+R_37; R_66 = 0.25*R_37;\n";
                os << "    R_66 = R_13+R_66; R_66 = R_9*R_66; R_66 = R_78+R_66; R_66 = R_0*R_66;\n";
                os << "    R_66 = R_16+R_66; R_16 = R_6*R_66; R_16 = 0.5*R_16; R_80 = R_16+R_80;\n";
                os << "    R_80 = 4.0*R_80; R_80 = R_31+R_80; R_80 = R_6*R_80; R_80 = 2.0*R_80;\n";
                os << "    R_77 = R_80+R_77; R_77 = 14.0*R_77; R_8 = R_77+R_8; R_66 = 24.0*R_66;\n";
                os << "    R_73 = R_66+R_73; R_47 = R_73+R_47; R_47 = 14.0*R_47; R_22 = R_47+R_22;\n";
                os << "    R_22 = R_7*R_22; R_37 = 10.0*R_37; R_32 = R_37+R_32; R_61 = R_32+R_61;\n";
                os << "    R_61 = 24.0*R_61; R_82 = R_61+R_82; R_59 = R_82+R_59; R_59 = 14.0*R_59;\n";
                os << "    R_59 = R_71+R_59; R_59 = R_5*R_59; R_39 = (-43589145600.0)*R_39; R_68 = R_39+R_68;\n";
                os << "    R_0 = pow(R_0,6); R_68 = R_0*R_68; R_0 = " << coeff_val << "; R_40 = R_0*R_40;\n";
                os << "    R_40 = (1.0/24.0)*R_40; R_10 = R_0*R_10; R_3 = R_10*R_3; R_3 = (-1.0/32768.0)*R_3;\n";
                os << "    R_40 = R_3+R_40; R_69 = R_0*R_69; R_69 = (1.0/1920.0)*R_69; R_40 = R_69+R_40;\n";
                os << "    R_67 = R_0*R_67; R_67 = (1.0/322560.0)*R_67; R_40 = R_67+R_40; R_8 = R_0*R_8;\n";
                os << "    R_8 = (1.0/92897280.0)*R_8; R_40 = R_8+R_40; R_22 = R_0*R_22; R_22 = (1.0/40874803200.0)*R_22;\n";
                os << "    R_40 = R_22+R_40; R_59 = R_0*R_59; R_59 = (1.0/25505877196800.0)*R_59; R_40 = R_59+R_40;\n";
                os << "    R_68 = R_0*R_68; R_68 = (1.0/21424936845312000.0)*R_68; R_40 = R_68+R_40; result = R_40;\n";
                os << "    return result;\n";
                break;
            default:
                TODO;
            }
        }
        os << "};\n";
    } else {
        os << "    static const std::vector<std::pair<TF,std::array<TF," << degp + 1 << ">>> coeffs = {\n";
        for( std::size_t num_cut = 0; num_cut < coeffs.cuts.size(); ++num_cut ) {
            os << "        { " << std::setprecision( 10 ) << std::setw( 16 ) << coeffs.cuts[ num_cut ].position << ", { ";
            for( std::size_t i = 0; i < coeffs.cuts[ num_cut ].coeffs.size(); ++i )
                os << ( i ? ", " : "" ) << std::setprecision( 10 ) << std::setw( 16 ) << coeffs.cuts[ num_cut ].coeffs[ i ];
            os << " } },\n";
        }
        os << "    };\n";
        os << "    \n";
        os << "    return _r_polynomials_integration( coeffs );\n";
    }
}
