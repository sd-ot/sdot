#include "Inst.h"

namespace Symbolic {

Inst::Inst( Context *context ) : context( context ) {
}

Inst::~Inst() {
}

Inst *Inst::simplify() {
    return this;
}

}
