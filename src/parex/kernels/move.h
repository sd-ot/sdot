#include <parex/support/ERROR.h>
#include <parex/TaskRef.h>
using namespace parex;

template<class T>
T *move( Task *t, T & ) {
    if ( ! t->move_arg( 0 ) )
        ERROR( "no owned" );
    return nullptr;
}
