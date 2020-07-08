#include "Context.h"
#include "Number.h"
#include "BinOp.h"
#include "UnaOp.h"

namespace Symbolic {

Expr::Expr( const Rational &value ) : value( value ), inst( nullptr ) {
}

Expr::Expr( Inst *inst ) : inst( inst ) {
}

Expr::Expr( int value ) : Expr( Rational( value ) ) {
}

void Expr::write_to_stream( std::ostream &os ) const {
    if ( inst )
        inst->write_to_stream( os );
    else
        os << "0";
}

void Expr::simplify() {
    if ( inst )
        inst = inst->simplify();
}

Expr &Expr::operator+=( const Expr &that ) { *this = *this + that; return *this; }
Expr &Expr::operator-=( const Expr &that ) { *this = *this - that; return *this; }
Expr &Expr::operator*=( const Expr &that ) { *this = *this * that; return *this; }
Expr &Expr::operator/=( const Expr &that ) { *this = *this / that; return *this; }

Expr Expr::operator-() const {
    if ( ! inst )
        return - value;
    return una_op( "-", *this );
}

Symbolic::Expr::operator bool() const {
    return Rational( *this );
}

Symbolic::Expr::operator Rational() const {
    if ( inst ) {
        if ( Number *n = dynamic_cast<Number *>( inst ) )
            return n->value;
        TODO;
    }
    return value;
}

Expr bin_op( std::string name, Expr a, Expr b ) {
    if ( ! a.inst && b.inst )
        return bin_op( name, b.inst->context->number( a.value ), b );
    if ( a.inst && ! b.inst )
        return bin_op( name, a, a.inst->context->number( b.value ) );
    if ( ! a.inst && ! b.inst ) {
        if ( name == "+" ) return a.value + b.value;
        if ( name == "-" ) return a.value - b.value;
        if ( name == "*" ) return a.value * b.value;
        if ( name == "/" ) return a.value / b.value;
        if ( name == "<" ) return a.value < b.value;
        TODO;
    }

    Inst *ai = a.inst, *bi = b.inst;
    if ( ( name == "+" || name == "*" ) && ai > bi )
        std::swap( ai, bi );
    for( Inst *p : a.inst->parents )
        if ( BinOp *b = dynamic_cast<BinOp *>( p ) )
            if ( b->name == name && b->children[ 0 ] == ai && b->children[ 1 ] == bi )
                return p;
    return a.inst->context->pool.create<BinOp>( a.inst->context, name, ai, bi );
}

Expr una_op( std::string name, Expr a ) {
    if ( ! a.inst ) {
        if ( name == "-" ) return - a.value;
        TODO;
    }

    for( Inst *p : a.inst->parents )
        if ( UnaOp *b = dynamic_cast<UnaOp *>( p ) )
            if ( b->name == name && b->children[ 0 ] == a.inst )
                return p;
    return a.inst->context->pool.create<UnaOp>( a.inst->context, name, a.inst );
}

Expr operator+( Expr a, Expr b ) { return bin_op( "+", a, b ); }
Expr operator-( Expr a, Expr b ) { return bin_op( "-", a, b ); }
Expr operator*( Expr a, Expr b ) { return bin_op( "*", a, b ); }
Expr operator/( Expr a, Expr b ) { return bin_op( "/", a, b ); }
Expr operator<( Expr a, Expr b ) { return bin_op( "<", a, b ); }

}
