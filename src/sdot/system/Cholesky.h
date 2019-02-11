#pragma once

template<class TF>
void cholesky( TF *L, const TF *A, int n ) {
    for( int i = 0; i < n; ++i ) {
        for( int j = 0; j <= i; ++j ) {
            TF s = 0;
            for( int k = 0; k < j; ++k )
                s += L[ i * ( i + 1 ) / 2 + k ] * L[ j * ( j + 1 ) / 2 + k ];
            L[ i * ( i + 1 ) / 2 + j ] = 1.0 / L[ j * ( j + 1 ) / 2 + j ] * ( A[ i * ( i + 1 ) / 2 + j ] - s );
        }
        TF s = 0;
        for( int k = 0; k < i; ++k )
            s += std::pow( L[ i * ( i + 1 ) / 2 + k ], 2 );
        L[ i * ( i + 3 ) / 2 ] = std::sqrt( A[ i * ( i + 3 ) / 2 ] - s );
    }
}

template<class TF>
void solve_using_cholesky( TF *res, const TF *chol, const TF *vec, int n ) {
    for( int r = 0; r < n; ++r ) {
        TF v = vec[ r ];
        for( int c = 0; c < r; ++c )
            v -= chol[ r * ( r + 1 ) / 2 + c ] * res[ c ];
        res[ r ] = v / chol[ r * ( r + 3 ) / 2 ];
    }

    for( int r = n; r--; ) {
        TF v = res[ r ];
        for( int c = r + 1; c < n; ++c )
            v -= chol[ c * ( c + 1 ) / 2 + r ] * res[ c ];
        res[ r ] = v / chol[ r * ( r + 3 ) / 2 ];
    }
}
