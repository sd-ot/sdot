#include "../Support/Stream.h"
#include "FunctionEnum.h"

namespace sdot {
namespace FunctionEnum {

void Arf::make_approximations_if_not_done() const {
    if ( approximations.size() )
        return;
    mutex.lock();
    if ( approximations.size() ) {
        mutex.unlock();
        return;
    }
    
    approximations.push_back( Approximation{} );
    P( "new approx" );

    mutex.unlock();
}

}
}
