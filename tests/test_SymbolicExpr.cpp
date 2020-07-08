#include "../src/sdot/support/symbolic/Codegen.h"
#include "../src/sdot/support/P.h"
using namespace Symbolic;

int main() {
    Context ctx;
    Expr a = ctx.named( "a" );
    Expr b = ctx.named( "b" );

    Codegen cg;
    cg.add_expr( "return", ( a - b ) * ( a + b ) );
    cg.write( std::cout );
}
