#pragma once

#include "../../src/sdot/support/OptParm.h"
#include <functional>
#include <fstream>
#include <bitset>

/**
*/
class GenPlaneCutProc {
public:
    /**/        GenPlaneCutProc    ( OptParm &op, std::string scalar_type, std::string size_type, std::string arch );

    void        gen                ( std::ostream &os );

    void        gen_sp_and_case    ( std::ostream &os );
    void        gen_header         ( std::ostream &os );
    void        gen_footer         ( std::ostream &os );
    void        gen_cases          ( std::ostream &os );
    void        gen_store          ( std::ostream &os, int s );
    void        gen_case           ( std::ostream &os, std::bitset<32> case_code );
    void        gen_load           ( std::ostream &os );

    void        for_each_reg       ( std::function<void( int off, int len )> f, int size );
    void        for_each_reg       ( std::function<void( int off, int len )> f );
    int         scalar_size        ();
    int         simd_size          ();
    std::string fv                 ( int len );
    std::string sv                 ( int len );

    int         size_for_test_during_load;
    std::string scalar_type;
    std::string size_type;
    std::string arch;
    int         size;          ///< nb scalars to handle with registers
    OptParm    &op;
};
