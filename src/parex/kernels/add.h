#include <parex/TaskRef.h>
using namespace parex;

template<class T>
T *add( Task *t, T &a, T &b ) {
    if ( t->move_arg( 0 ) )
        return a += b, nullptr;
    if ( t->move_arg( 1 ) )
        return b += a, nullptr;
    return new T( a + b );
}
