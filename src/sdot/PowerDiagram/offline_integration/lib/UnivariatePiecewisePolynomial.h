#pragma once

#include "../../system/Stream.h"

template<class TF>
class UnivariatePiecewisePolynomial {
public:
    struct Cut  {
        void            write_to_stream( std::ostream &os ) const { os << position << ": [" << coeffs << "]"; }

        TF              position;
        std::vector<TF> coeffs;
        std::size_t     index;
    };

    void write_to_stream( std::ostream &os ) const {
        os << cuts;
    }

    TF operator()( TF x ) const {
        // TODO: binary search
        for( std::size_t nc = 0; ; ++nc ) {
            if ( nc == cuts.size() - 1 || x <= cuts[ nc ].position ) {
                TF res = 0;
                for( std::size_t i = 0; i < cuts[ nc ].coeffs.size(); ++i )
                    res += cuts[ nc ].coeffs[ i ] * pow( x, i );
                return res;
            }
        }
        return 0;
    }

    std::vector<Cut> cuts;
};
