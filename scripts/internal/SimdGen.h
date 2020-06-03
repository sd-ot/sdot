#pragma once

#include "SimdGenOp.h"
#include <functional>

/***/
class SimdGen {
public:
    struct             Write            { std::string name; SimdGenOp::ST nop; };
    using              Op               = SimdGenOp;
    using              ST               = Op::ST;

    void               add_write        ( std::string name, ST nop );

    ST                 new_undefined    ( ST len, std::string type );
    ST                 new_gather       ( std::vector<ST> nops, std::vector<ST> nouts );
    ST                 new_add          ( ST a, ST b ) { return new_bop( Op::Type::Add, a, b ); }
    ST                 new_sub          ( ST a, ST b ) { return new_bop( Op::Type::Sub, a, b ); }
    ST                 new_mul          ( ST a, ST b ) { return new_bop( Op::Type::Mul, a, b ); }
    ST                 new_div          ( ST a, ST b ) { return new_bop( Op::Type::Div, a, b ); }
    ST                 new_bop          ( Op::Type type, ST a, ST b );
    ST                 new_var          ( std::string name, ST len, std::string scalar_type );
    ST                 new_op           ( Op::Type type, ST len, std::string scalar_type );

    bool               all_children_done( const std::vector<bool> &done, ST nop );
    void               handle_undefined ();
    void               for_each_child   ( std::function<void( ST nop )> f, std::vector<bool> &seen, ST nop );
    void               for_each_child   ( std::function<void( ST nop )> f );
    void               write_inst       ( std::ostream &os, std::string sp, std::vector<std::string> &tmps, ST nop );
    void               gen_code         ( std::ostream &os, std::string sp );

    std::string        arch             = "AVX2";

private:
    void               update_parents   ();

    std::vector<Write> writes;
    std::vector<Op>    ops;
};
