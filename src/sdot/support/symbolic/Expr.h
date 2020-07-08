#pragma once

#include "../Rational.h"

namespace Symbolic {
class Context;
class Inst;

/**
*/
class Expr {
public:
    /**/              Expr           ( const Rational &value = 0 );
    /**/              Expr           ( Inst *inst );
    /**/              Expr           ( int value );

    void              write_to_stream( std::ostream &os ) const;
    void              simplify       ();

    Expr&             operator+=     ( const Expr &that );
    Expr&             operator-=     ( const Expr &that );
    Expr&             operator*=     ( const Expr &that );
    Expr&             operator/=     ( const Expr &that );
    Expr              operator-      () const;

    explicit operator Rational       () const;
    explicit operator bool           () const;

    Rational          value;
    Inst*             inst;
};

Expr bin_op( std::string name, Expr a, Expr b );
Expr una_op( std::string name, Expr a );

Expr operator+( Expr a, Expr b );
Expr operator-( Expr a, Expr b );
Expr operator*( Expr a, Expr b );
Expr operator/( Expr a, Expr b );

Expr operator<( Expr a, Expr b );

}
