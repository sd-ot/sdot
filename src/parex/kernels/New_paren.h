#include <parex/support/S.h>
#include <utility>

template<class T,class... Args>
T *New_paren( parex::S<T>, Args&& ...args ) {
    return new T( std::forward<Args>( args )... );
}
